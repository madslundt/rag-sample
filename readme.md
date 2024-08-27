
# Overview
This project provides a set of Python scripts to populate and query a vectorstore database using LangChain's functionalities. The main features include:

- Database Population: `populate_database.py` is used to load documents from a PDF file, split them into manageable chunks, and add them to a Chroma vectorstore. This process utilizes embeddings, which are numerical representations of text that capture semantic meaning, allowing efficient similarity search and retrieval of relevant content.
- Interactive Querying: `query_data.py` provides an interface to query the database interactively or via command-line arguments. It uses a Large Language Model (LLM) with multiquery capabilities to generate alternative versions of user queries, enhancing the retrieval of relevant documents and generating comprehensive answers.
- Testing: Basic tests are provided to validate the querying capabilities using `pytest`.

# Prerequisites
- Python 3.11
- pipenv: Required to manage dependencies
- Ollama: Required to run the LLM and embeddings locally. Ensures that language model and an embedding function are accessible for processing queries and documents.

By default, `llama3.1` is used as the LLM, and `nomic-embed-text` is utilized as the embedding function (these settings can be modified in the `env.py` file).

# Installation
1. Clone the repository to your local machine.

```bash
git clone https://github.com/madslundt/rag-sample
cd rag-sample
```

2. Install the required dependencies using `pipenv`.

```bash
pipenv install
```

3. Activate the virtual environment.

```bash
pipenv shell
```

# Usage
## 1. Populating the Database
To populate the Chroma vectorstore database with the `Owners_Manual.pdf`, use the `populate_database.py` script. You can reset the database before loading new documents using the `--reset` flag.
The embedding process converts text into numerical vectors for efficient similarity search. Each document chunk is assigned a unique `id` and a `hash` value to ensure that duplicates are not added. The script checks these identifiers against the existing database entries to determine if the document is already present.

```bash
python populate_database.py --reset
```

## 2. Querying the Database
You can query the database using `query_data.py` either by providing a query directly as a command-line argument or interactively.
This script uses the LLM to generate multiple versions of the input query, improving the retrieval of relevant information and generating a well-informed response.
The sources of the retrieved information are displayed, which are extracted from the metadata of the documents stored in the vectorstore.

To query using the command line:

```bash
python query_data.py --query_text "Your query here"
```

3. To enter interactive mode:

```bash
python query_data.py
```

In interactive mode, you can type your queries directly, and type `exit` or `q` to quit.

## 3. Running Tests
To run tests and validate the querying logic, use `pytest`.

```bash
pytest
```

# Project Structure
- `populate_database.py`: Script for clearing, loading, splitting, and adding documents to the vectorstore.
- `query_data.py`: Script for querying the vectorstore database and generating responses based on user questions using the LLM.
- `test_rag.py`: Test cases to verify the functionality of the scripts by using an LLM to check if the results are accurate..
- `env.py`: Contains environment variables and configurations such as paths and model names.
- `helpers.py`: Helper functions for verbose printing and other utility tasks.
- `get_vectorstore.py`: Script to initialize and get access to the vectorstore.

# Notes
The database is cleared when running `populate_database.py` with the `--reset` flag.
Customize the paths and configurations in env.py according to your project setup.

# License
This project is licensed under the terms of the MIT license.

# Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.
