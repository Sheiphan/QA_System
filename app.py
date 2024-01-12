import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
import re, os

class DataLoader:
    def __init__(self, path):
        self.path = path

    def get_documents(self):
        """
        The function `get_documents` loads documents from a directory using a `DirectoryLoader` object.
        :return: The method `get_documents` returns the result of calling the `load` method on an
        instance of the `DirectoryLoader` class.
        """
        loader = DirectoryLoader(self.path)
        return loader.load()

    def divide_documents(self, documents, size=1000, overlap=20):
        """
        The function divides a list of documents into smaller chunks based on a specified size and
        overlap.
        
        :param documents: The `documents` parameter is a list of documents that you want to divide into
        smaller chunks. Each document can be a string or any other data type that represents a document
        :param size: The size parameter determines the maximum number of characters in each chunk of
        text. If a document exceeds this size, it will be split into multiple chunks, defaults to 1000
        (optional)
        :param overlap: The overlap parameter determines the number of characters that will be shared
        between adjacent chunks. In other words, if the overlap is set to 20, the last 20 characters of
        one chunk will be included in the next chunk. This can help ensure that important information is
        not split between chunks, defaults to 20 (optional)
        :return: the result of calling the `split_documents` method of the `splitter` object.
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=size, 
                                                  chunk_overlap=overlap)
        return splitter.split_documents(documents)


class VectorDatabase:
    def __init__(self):
        self.embedding = SentenceTransformerEmbeddings()

    def generate_chroma(self, documents):
        """
        The function generates chroma vectors using FAISS from a given set of documents and an
        embedding.
        
        :param documents: The "documents" parameter is a list of text documents that you want to
        generate chroma vectors for. Each document should be a string
        :return: the result of the `FAISS.from_documents` method, which is likely a representation of
        the documents in a high-dimensional space based on their embeddings.
        """
        return FAISS.from_documents(documents, 
                                    self.embedding)


class LLMChain:
    def __init__(self, model):
        self.llm = ChatOpenAI(model_name=model, 
                              openai_api_key=st.secrets["OPENAI_API_KEY"], 
                              temperature=0.5)

    def execute_chain(self, qa_chain, question, docs):
        """
        The function executes a QA chain on a given question and set of documents.
        
        :param qa_chain: The `qa_chain` parameter is an instance of a question answering model or
        pipeline. It is responsible for processing the input documents and generating an answer to the
        given question
        :param question: The question parameter is a string that represents the question you want to ask
        :param docs: The `docs` parameter is a list of documents that the QA chain will use to find the
        answer to the question. Each document in the list should be a string representing the text of
        the document
        :return: The function `execute_chain` returns the result of running the `qa_chain` on the given
        `docs` and `question`.
        """
        return qa_chain.run(input_documents=docs, 
                            question=question)

def display_result(response):
    st.write(response)


def get_user_input(prompt):
    return f"{st.text_input('Input:')}. {prompt}"

def find_agents_button():
    return st.button("Find Agents")


def run_app():
    """
    The `run_app` function takes user input, searches for matching documents, and uses a language model
    to generate a response based on the user query and the matching documents.
    """
    st.title("AI Challenge Application Demo | QA System")

    prompt = (
        "Assume you are a senior consultant to a Police Department (PD). "
        "Find two agents that are best suited for the task, consider the skills and restrictions that they have."
        "Answer with reason. Keep the answer to the point and precise."
    )
    user_query = get_user_input(prompt)

    if find_agents_button():
        doc_path = "data"
        
        try:
            doc_manager = DataLoader(doc_path)
            loaded_docs = doc_manager.get_documents()
            segmented_docs = doc_manager.divide_documents(loaded_docs)
        except Exception as e:
            st.error(f"Error processing documents: {e}")
            return
        
        embedding_mgr = VectorDatabase()
        try:
            chroma_db = embedding_mgr.generate_chroma(segmented_docs)
            matching_documents = chroma_db.similarity_search(user_query)
        except Exception as e:
            st.error(f"Error generating chroma vectors or searching for similar documents: {e}")
            return

        ai_chain = LLMChain("gpt-3.5-turbo")
        if ai_chain.llm is None:
            st.error("Error initializing ChatOpenAI. Check your API key.")
            return

        qa_chain = load_qa_chain(ai_chain.llm, chain_type="stuff", verbose=False)
        if qa_chain is None:
            st.error("Error loading QA chain.")
            return

        try:
            response = ai_chain.execute_chain(qa_chain, user_query, matching_documents)
            display_result(response)
        except Exception as e:
            st.error(f"Error executing QA chain: {e}")

    
if __name__ == "__main__":
    run_app()
