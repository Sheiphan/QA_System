# AI Challenge Application Demo | QA System

This application demonstrates a Question-Answering (QA) system using language models and document similarity. The system is designed to find agents suited for a given task based on user input and contextual documents.

## Overview

The QA system utilizes the following components:

- **DataLoader**: Loads documents from a directory using `DirectoryLoader` and provides a method to divide documents into smaller chunks using `RecursiveCharacterTextSplitter`.

- **VectorDatabase**: Generates chroma vectors using FAISS from a set of text documents and an embedding provided by `SentenceTransformerEmbeddings`.

- **LLMChain**: Executes a QA chain on a given question and set of documents using `ChatOpenAI` and `load_qa_chain`.

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/Sheiphan/QA_System.git
   cd your_repository
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment by creating a `.env` file with your OpenAI API key:

   ```
   OPENAI_KEY=your_openai_api_key
   ```

4. Run the application:

   ```bash
   streamlit run app.py
   ```

5. Enter your task in the provided input box and click the "Find Agents" button to get the system's response.

## Requirements

- Python 3.x
- Streamlit
- Other dependencies specified in `requirements.txt`


## Acknowledgments

- [Streamlit](https://streamlit.io/) for the web application framework.
- [LangChain](https://github.com/langchain/langchain) for the language models and processing components.
- [SentenceTransformer](https://www.sbert.net/) for providing pre-trained embeddings.
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search.

## Author

Sheiphan Joseph
