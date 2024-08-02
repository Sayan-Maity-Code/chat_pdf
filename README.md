# Advanced Document Q&A System

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [System Architecture](#system-architecture)
6. [Dependencies and Open-Source Models](#dependencies-and-open-source-models)
7. [Process Flow](#process-flow)
8. [Limitations](#limitations)
9. [Future Improvements](#future-improvements)
10. [Note on Model Usage](#note-on-model-usage)
11. [Custom CSS](#custom-css)

## Introduction

The Advanced Document Q&A System is a sophisticated tool that integrates document processing, natural language understanding, and machine learning to offer an interactive question-answering experience based on document content. This system processes uploaded documents, extracts key information, and enables users to ask questions about the content.

## Features

- **Document Parsing**: Extract text and metadata using LlamaParse.
- **Text Chunking**: Split text into manageable chunks with RecursiveCharacterTextSplitter.
- **Embedding Generation**: Create text embeddings using HuggingFace models.
- **Similarity Search**: Efficiently search and cluster text chunks using FAISS (Facebook AI Similarity Search).
- **Question Answering**: Provide contextually relevant answers using FLAN-T5.
- **Document Summarization**: Generate summaries of entire documents.
- **User Interface**: Interactive and user-friendly interface built with Gradio.

## Installation

1. **Clone the Repository**:
    ```bash
    git clone [your-repository-url]
    cd [your-project-directory]
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Set Up Environment Variables**:
    Create a `.env` file in the project root and add your LlamaParse API key:
    ```plaintext
    llama_parser=your_api_key_here
    ```

4. **Download Model Files**:
    - **SentenceTransformer**: `sentence-transformers/all-MiniLM-L6-v2`
    - **FLAN-T5**: `google/flan-t5-large`

## Usage

1. **Run the Gradio App**:
    ```bash
    python [your_main_script_name].py
    ```

2. **Interact with the Application**:
    - Open your web browser and navigate to the URL provided by Gradio (usually http://localhost:7860).
    - **Upload and Process Documents**: Upload a document for processing.
    - **Ask Questions**: Input questions to get answers based on the document content.
    - **View Summaries**: Generate and view summaries of the document.

## System Architecture

The system is designed with a modular architecture, incorporating separate components for document processing, indexing, and question answering. It utilizes several open-source models and libraries for its functionality.

## Dependencies and Open-Source Models

- **Gradio**: For building the web-based user interface.
- **LlamaParse**: For document parsing and text extraction.
    - API Key required (1000 pages per day for free accounts).
    - Maximum 1200 pages per file.
- **FAISS**: For efficient similarity search and indexing of embeddings.
- **HuggingFace Transformers**:
    - **SentenceTransformer**: `sentence-transformers/all-MiniLM-L6-v2` for text embeddings.
    - **FLAN-T5**: `google/flan-t5-large` for text generation and question answering.
- **LangChain**: For building the question-answering chain and document processing pipeline.
- **python-dotenv**: For loading environment variables.
- **Other Python Libraries**: `os`, `shutil`

## Process Flow

1. **Document Processing**:
    - Use LlamaParse to extract text from the uploaded document.
    - Split the extracted text into smaller chunks using RecursiveCharacterTextSplitter.

2. **Indexing**:
    - Generate embeddings for text chunks using HuggingFaceEmbeddings.
    - Create an efficient index for embeddings using FAISS.

3. **Question Answering**:
    - Retrieve relevant text chunks using the FAISS index.
    - Refine the retrieved chunks with ContextualCompressionRetriever.
    - Generate answers using the FLAN-T5 model.

4. **Summarization**:
    - Generate a summary of the entire document using indexed content.

5. **Result Presentation**:
    - Display generated answers or summaries through the Gradio interface.

## Limitations

- **Document Size**: Processing large documents may be limited by system resources.
- **Processing Time**: Parsing and indexing large documents can be time-consuming.
- **API Limits**: LlamaParse API has usage limits (1000 pages per day for free accounts).
- **Answer Quality**: Depends on the accuracy of document parsing and relevance of context.
- **Language Support**: Optimized for English content.

## Future Improvements

- Implement multi-language support for document analysis and question answering.
- Optimize processing time for large documents through parallel processing.
- Enhance the retrieval system for better handling of context and document structure.
- Integrate advanced language models for improved answer generation.
- Add support for multiple document types and formats.
- Implement caching to improve response times for repeated queries.

## Note on Model Usage

The system uses several pre-trained models:

- **SentenceTransformer**: `all-MiniLM-L6-v2` for text embeddings.
- **CLIP**: `openai/clip-vit-base-patch32` for potential future image-related tasks.
- **FLAN-T5**: `google/flan-t5-large` for text generation and question answering.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

These models should be downloaded separately due to their size.

## Custom CSS

The project includes custom CSS for the Gradio interface, setting a background image and styling various elements for a more appealing user experience.
