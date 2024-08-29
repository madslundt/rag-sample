import argparse
from langchain.retrievers.multi_query import LineListOutputParser
from env import CHILD_CHUNK_SIZE, DOCSTORE_PATH, DOCSTORE_TABLE_NAME, OLLAMA_MODEL, PARENT_DOC_ID
from utils import get_sqlitestore, verbose_print
from utils.get_vectorstore import get_vectorstore
from langchain_core.prompts import PromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

USE_MULTIVECTOR_RETRIEVER: bool = CHILD_CHUNK_SIZE > 0

def main() -> None:
    """Main function to handle command-line arguments and interactive querying."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    if query_text:
        query_rag(query_text)
    else:
        interactive_query_loop()

def interactive_query_loop() -> None:
    """Runs an interactive loop to query until 'exit' or 'q' is entered."""
    while True:
        query = input("\nQuery: ").strip()
        if query.lower() in {"exit", "q"}:
            break
        if query:
            query_rag(query)

def get_prompt(template: str, input_variables: list[str]) -> PromptTemplate:
    """Creates a PromptTemplate with a given template and input variables."""
    return PromptTemplate(template=template, input_variables=input_variables)

def get_metadata_field_info() -> list[AttributeInfo]:
    """Returns metadata field information."""
    return [
        AttributeInfo(
            name="id",
            description="Unique id of the document chunk",
            type="string",
        ),
        AttributeInfo(
            name="source",
            description="Source name of the document from which the information was extracted.",
            type="string",
        ),
        AttributeInfo(
            name="hash",
            description="SHA1 Hash of the document chunks",
            type="string",
        ),
        AttributeInfo(
            name="page",
            description="Page the chunk appears on in the main document",
            type="int",
        )
    ]

def query_rag(query_text: str) -> None:
    """Handles the query process, from generating alternatives to retrieving relevant documents and generating a response."""
    try:
        vectorstore = get_vectorstore()
        retriever: BaseRetriever
        if USE_MULTIVECTOR_RETRIEVER:
            docstore = get_sqlitestore(DOCSTORE_PATH, DOCSTORE_TABLE_NAME)
            retriever = MultiVectorRetriever(
                vectorstore=vectorstore,
                docstore=docstore,
                id_key=PARENT_DOC_ID,
            )
        else:
            retriever = vectorstore.as_retriever()

        llm = Ollama(model=OLLAMA_MODEL)
        query_output_parser = LineListOutputParser()

        query_prompt = get_prompt(
            template="""You are an AI language model assistant. Your task is to generate five
            different versions of the given user question to retrieve relevant documents from a vector
            database. By generating multiple perspectives on the user question, your goal is to help
            the user overcome some of the limitations of the distance-based similarity search.
            Provide these and only these alternative questions separated by newlines.
            Original question: {question}""",
            input_variables=["question"]
        )

        model = query_prompt | llm | query_output_parser
        questions = model.invoke({"question": query_text})[1:]

        verbose_print("\n".join(questions), "\n")

        relevant_docs, source_pages = retrieve_relevant_docs(questions, retriever)
        response_text = generate_response(query_text, relevant_docs)

        print(f"Response: {response_text}\nSources: {source_pages}")
        return response_text

    except Exception as e:
        print(f"An error occurred: {e}")

def retrieve_relevant_docs(questions: list[str], retriever: BaseRetriever) -> tuple[list, list]:
    """Retrieves relevant documents based on generated questions."""
    relevant_docs = []
    source_ids = set()
    source_pages = []

    for search in questions:
        _relevant_docs = [
            doc for doc in retriever.invoke(search) if doc.metadata.get("id") not in source_ids
        ]
        relevant_docs.extend(_relevant_docs)
        source_ids.update(doc.metadata.get("id") for doc in _relevant_docs)
        source_pages.extend(f"{doc.metadata.get('source', 'unknown')} page {doc.metadata.get('page', 'unknown')}" for doc in _relevant_docs)

    return relevant_docs, source_pages

def generate_response(query_text: str, relevant_docs: list[Document]) -> str:
    """Generates a response based on the context from relevant documents."""
    context_text = "\n\n---\n\n".join(doc.page_content for doc in relevant_docs)
    prompt = get_prompt(
        template="""Answer the question based only on the following context:

        {context}

        ---

        Answer the question based on the above context: {question}""",
        input_variables=["context", "question"]
    )

    llm = Ollama(model=OLLAMA_MODEL)
    model = prompt | llm
    response_text = model.invoke({"context": context_text, "question": query_text})
    return response_text

if __name__ == "__main__":
    main()
