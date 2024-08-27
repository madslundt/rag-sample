import argparse
import hashlib
import os
import shutil
from env import CHROMA_PATH
from get_vectorstore import get_vectorstore
from helpers import verbose_print
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

def main():
    """Main function to handle database reset and document processing."""
    args = parse_arguments()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Load, split, and add documents to the database
    print("\n\n------\nLoading documents\n------\n\n")
    documents = load_documents()

    print("\n\n------\nSplitting documents\n------\n\n")
    chunks = split_documents(documents)

    print("\n\n------\nAdding documents to Chroma\n------\n\n")
    add_to_chroma(chunks)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    return parser.parse_args()

def load_documents():
    """Load documents from PDF using PyPDFLoader."""
    loader = PyPDFLoader("Owners_Manual.pdf")
    return loader.load_and_split()

def split_documents(documents):
    """Split documents into chunks using RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def chunk_list(lst, chunk_size):
    """Divide a list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def add_to_chroma(chunks, chunk_size=500):
    """Add or update document chunks in the Chroma vectorstore."""
    db = get_vectorstore()
    chunks_with_ids = generate_chunks_with_metadata(chunks)

    existing_ids = set(db.get(include=[])["ids"])
    verbose_print(f"\tNumber of existing documents in DB: {len(existing_ids)}")

    new_chunks, updated_chunks = partition_chunks(chunks_with_ids, existing_ids, db)

    process_new_chunks(new_chunks, db, chunk_size)
    process_updated_chunks(updated_chunks, db, chunk_size)

def partition_chunks(chunks_with_ids, existing_ids, db):
    """Partition chunks into new and updated based on existing IDs and hash."""
    new_chunks = []
    updated_chunks = []

    for chunk in chunks_with_ids:
        chunk_id = chunk.metadata["id"]
        chunk_hash = chunk.metadata["hash"]

        if chunk_id not in existing_ids:
            new_chunks.append(chunk)
        else:
            existing_doc = db.get(ids=[chunk_id])
            if existing_doc and existing_doc["metadatas"][0].get("hash") != chunk_hash:
                updated_chunks.append(chunk)

    return new_chunks, updated_chunks

def process_new_chunks(new_chunks, db, chunk_size):
    """Process and add new chunks to the database."""
    if new_chunks:
        verbose_print(f"\t👉 Adding new documents: {len(new_chunks)}")
        for idx, chunk_group in enumerate(chunk_list(new_chunks, chunk_size)):
            db.add_documents(chunk_group, ids=[chunk.metadata["id"] for chunk in chunk_group])
            verbose_print(f"\t👉 Added: {chunk_size * idx + len(chunk_group)}")
    else:
        verbose_print("\t✅ No new documents to add")

def process_updated_chunks(updated_chunks, db, chunk_size):
    """Process and update existing chunks in the database."""
    if updated_chunks:
        verbose_print(f"\t👉 Updating documents: {len(updated_chunks)}")
        for idx, chunk_group in enumerate(chunk_list(updated_chunks, chunk_size)):
            db.add_documents(chunk_group, ids=[chunk.metadata["id"] for chunk in chunk_group])
            verbose_print(f"\t👉 Updated: {chunk_size * idx + len(chunk_group)}")
    else:
        verbose_print("\t✅ All documents are up-to-date")

def generate_hash(text):
    """Generate SHA-256 hash for the given text."""
    return hashlib.sha256(text.encode()).hexdigest()

def generate_chunks_with_metadata(chunks):
    """Generate metadata for document chunks, including unique IDs and hash."""
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page or 0}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        chunk.metadata["id"] = chunk_id
        chunk.metadata["hash"] = generate_hash(chunk.page_content)
        last_page_id = current_page_id

    return chunks

def clear_database():
    """Clear the Chroma database."""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()
