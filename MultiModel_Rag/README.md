# MultiModel RAG Project

This project implements a MultiModel Retrieval-Augmented Generation (RAG) pipeline using LangChain and other related libraries. The pipeline supports text processing, embeddings, document retrieval, and integration with large language models (LLMs) for advanced query-answering capabilities. Additionally, the project includes utilities for PDF parsing, text splitting, and embeddings generation.

## Features!

- **PDF Handling**: Extracts text from PDFs using PyMuPDF (`fitz`).
- **Embeddings**: Leverages HuggingFace embeddings for semantic representation.
- **Document Storage**: Uses in-memory storage for managing documents.
- **Vector Search**: Employs Chroma for multi-vector document retrieval.
- **LLM Integration**: Uses the Ollama LLM interface for query answering.
- **Utilities**: Includes functions for image handling, progress visualization, and text splitting.

## Prerequisites

Ensure you have the following installed before setting up the project:

- Python 3.8 or higher

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd MultiModel_Rag
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the required HuggingFace model:
   ```bash
   from langchain.embeddings import HuggingFaceEmbeddings
   embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-large-en')
   ```

## Usage

1. **PDF Parsing**:
   Use the following code to extract text and tables from a PDF file:
   ```python
   import fitz

   def extract_text_and_tables(pdf_path):
       doc = fitz.open(pdf_path)
       text = ""
       for page in doc:
           text += page.get_text("text")
       return text

   pdf_path = "example.pdf"
   text = extract_text_and_tables(pdf_path)
   print(text)
   ```

2. **Embeddings Creation**:
   ```python
   from langchain.embeddings import HuggingFaceEmbeddings
   embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-large-en')
   ```

3. **RAG Pipeline**:
   Build a Retrieval-Augmented Generation pipeline with LangChain:
   ```python
   from langchain_chroma import Chroma
   from langchain.retrievers.multi_vector import MultiVectorRetriever

   # Example Chroma setup
   chroma = Chroma(persist_directory="./chroma_storage", embedding_function=embeddings)
   retriever = MultiVectorRetriever(store=chroma)
   ```

4. **Query the Model**:
   ```python
   from langchain_ollama import OllamaLLM

   llm = OllamaLLM(model="llama2")
   query = "What is RAG?"
   response = llm(query)
   print(response)
   ```

## File Structure

- `requirements.txt`: Lists all the dependencies.
- `MultiModel_Rag.ipynb`: Jupyter Notebook for the entire pipeline.
- `README.md`: Documentation for the project.

## Dependencies

See `requirements.txt` for the full list of dependencies!

## Contributing

Feel free to submit issues or pull requests for enhancements or bug fixes.
