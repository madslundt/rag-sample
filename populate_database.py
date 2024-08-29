import argparse
import hashlib
import os
import shutil
from typing import Optional
from env import CHROMA_PATH, DOCSTORE_PATH, DOCSTORE_TABLE_NAME, PARENT_CHUNK_SIZE, PARENT_DOC_ID, CHILD_CHUNK_SIZE
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_chroma.vectorstores import VectorStore
from utils import get_sqlitestore, get_vectorstore, verbose_print


def main() -> None:
    """Main function to handle database reset and document processing."""
    args = parse_arguments()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Load, split, and add documents to the database
    print("\n\n------\nLoading documents\n------\n\n")
    documents = load_documents()

    print("\n\n------\nSplitting documents\n------\n\n")
    docs, sub_docs = split_documents(documents, parent_chunk_size=PARENT_CHUNK_SIZE, child_chunk_size=CHILD_CHUNK_SIZE)

    print("\n\n------\nAdding documents to Chroma\n------\n\n")
    add_documents_to_store(docs, sub_docs)


def parse_arguments() -> None:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    return parser.parse_args()


def load_documents() -> None:
    """Load documents from PDF using PyPDFLoader."""
    loader = PyPDFLoader("Owners_Manual.pdf")
    return loader.load_and_split()


def split_documents(
    documents: list[Document],
    parent_chunk_size: int = 400,
    child_chunk_size: int = 0
) -> tuple[list[Document], list[Document]]:
    """
    Split documents into chunks and sub-chunks.
    if child_chunk_size is > 0, split the documents into sub-chunks as well.
    If child_chunk_size is 0, only split the documents into chunks.

    Args:
        documents (list[Document]): List of documents to split.
        parent_chunk_size (int, default 400): Size of parent chunks.
        child_chunk_size (int, default None): Size of child chunks.
    """
    parent_text_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_chunk_size)

    new_documents = generate_documents_with_metadata(parent_text_splitter.split_documents(documents))
    sub_documents = []

    child_text_splitter = None
    if child_chunk_size > 0:
        child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=child_chunk_size)

    for idx, document in enumerate(new_documents):
        if child_text_splitter:
            _sub_documents = generate_documents_with_metadata(
                child_text_splitter.split_documents([document]),
                idx
            )

            for sub_document in _sub_documents:
                sub_document.metadata[PARENT_DOC_ID] = document.metadata.get("id")
                sub_documents.append(sub_document)

    return new_documents, sub_documents


def chunk_list(lst: list, chunk_size: int) -> list[list]:
    """
    Divide a list into chunks of specified size.

    Args:
        lst (list): List to be chunked.
        chunk_size (int): Max size of each chunk.
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def add_documents_to_store(
        documents: list[Document],
        sub_documents: list[Document] = [],
        chunk_size: int = 500
) -> None:
    """
    Add documents to the vectorstore.
    if sub_documents is an empty list, then add the documents to the vectorstore.
    Else add sub_documents to the vectorestore and add documents to the bytestore.

    Args:
        documents (list[Document]): List of documents to add to the bytestore (if sub_documents is empty, then they'll be added to the vectorstore).
        sub_documents (list[Document]): List of sub-documents to add to the vectorstore.
        chunk_size (int, default 500): Number of documents to add in each batch.
    """
    vectorstore = get_vectorstore()
    vectorstore_documents = sub_documents if sub_documents else documents
    docstore = get_sqlitestore(DOCSTORE_PATH, DOCSTORE_TABLE_NAME)

    if sub_documents:
        docstore.mset(list(zip([doc.metadata["id"] for doc in documents], documents)))

    existing_ids = set(vectorstore.get(include=[])["ids"])
    verbose_print(f"\tNumber of existing documents in DB: {len(existing_ids)}")

    new_chunks, updated_chunks = get_documents_to_add_or_update(vectorstore_documents, existing_ids, vectorstore)

    if new_chunks:
        verbose_print(f"\t👉 Adding new documents: {len(new_chunks)}")
        add_or_update_documents_to_vectorstore(new_chunks, vectorstore, chunk_size)
    else:
        verbose_print("\t✅ No new documents to add")

    if updated_chunks:
        verbose_print(f"\t👉 Updating documents: {len(updated_chunks)}")
        add_or_update_documents_to_vectorstore(updated_chunks, vectorstore, chunk_size)
    else:
        verbose_print("\t✅ All documents are up-to-update")


def get_documents_to_add_or_update(
        documents: list[Document],
        existing_ids: list[str],
        vectorstore: VectorStore
) -> tuple[list[Document], list[Document]]:
    """
    Get documents that needs to be added (if the id is not present in the vectorstore) or updated (if the hash is different).
    Ignoring documents that are already present and have the same hash.

    Args:
        documents (list[Document]): List of documents with IDs.
        existing_ids (list[str]): List of existing document IDs in the vectorstore.
        vectorstore (VectorStore): Vectorstore instance.
    """
    new_documents = []
    updated_documents = []

    for document in documents:
        id = document.metadata["id"]
        hash = document.metadata["hash"]

        if id not in existing_ids:
            new_documents.append(document)
        else:
            existing_doc = vectorstore.get(ids=[id])
            if existing_doc and existing_doc["metadatas"][0].get("hash") != hash:
                updated_documents.append(document)

    return new_documents, updated_documents


def add_or_update_documents_to_vectorstore(
        documents: list[Document],
        vectorstore: VectorStore,
        chunk_size: int = 500
) -> None:
    """
    Add or update documents in the vectorstore.
    This is done in batches.

    Args:
        documents (list[Document]): List of documents to add or update.
        vectorstore (VectorStore): Vectorstore instance.
        chunk_size (int, default 500): Number of documents to add in each batch.
    """
    for idx, chunk_group in enumerate(chunk_list(documents, chunk_size)):
        vectorstore.add_documents(chunk_group, ids=[chunk.metadata["id"] for chunk in chunk_group])
        verbose_print(f"\t👉 Added: {chunk_size * idx + len(chunk_group)}")


def generate_hash(text: str) -> str:
    """
    Generate SHA-256 hash for the given text.

    Args:
        text (str): Text to generate hash for.
    """
    return hashlib.sha256(text.encode()).hexdigest()


def generate_documents_with_metadata(documents: list[Document], source_chunk_idx: Optional[int] = None) -> list[Document]:
    """
    Generate metadata for documents, including unique IDs and hash.

    Args:
        documents (list[Document]): List of documents to generate metadata for.
    """
    last_page_id = None
    current_chunk_index = 0

    for document in documents:
        source = document.metadata.get("source")
        page = document.metadata.get("page")
        current_page_id = f"{source}"
        if source_chunk_idx is not None:
            current_page_id += f":{source_chunk_idx}"

        current_page_id += f":{page or 0}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        id = f"{current_page_id}:{current_chunk_index}"
        document.metadata["id"] = id
        document.metadata["hash"] = generate_hash(document.page_content)
        last_page_id = current_page_id

    return documents


def clear_database() -> None:
    """
    Clear both chroma and docstore
    """
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    if os.path.exists(DOCSTORE_PATH):
        os.remove(DOCSTORE_PATH)


if __name__ == "__main__":
    main()
