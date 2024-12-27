import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as genai

# Load environment variables
load_dotenv()

genai_api_key = os.getenv("GOOGLE_API_KEY")

if not genai_api_key:
    st.error("Please set your GOOGLE_API_KEY in the .env file.")
else:
    genai.configure(api_key=genai_api_key)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

def get_pdf_text(pdf_docs):
    """
    Extract text from the uploaded PDF files.

    Args:
        pdf_docs (list): List of uploaded PDF files.

    Returns:
        str: Concatenated text from all PDF pages.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """
    Split the extracted text into smaller chunks for processing.

    Args:
        text (str): The extracted text from the PDF files.

    Returns:
        list: List of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """
    Create and save a FAISS vector store using the text chunks.

    Args:
        text_chunks (list): List of text chunks to be embedded and stored.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """
    Create a conversational QA chain with a custom prompt template.

    Returns:
        Chain: A QA chain object configured with Google Generative AI.
    """
    prompt_template = """
        You are a chatbot application. Please answer the user queries from the provided context. 
        If the answer is not available in the provided context, feel free to reply with answer not available prompt. 
        Do not make up answers.
        Context: \n {context}? \n
        Question: \n {question}\n
        Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    """
    Process the user input, perform similarity search, and generate a response.

    Args:
        user_question (str): The user's query.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    st.session_state["chat_history"].append({"role": "user", "content": user_question})
    st.session_state["chat_history"].append({"role": "assistant", "content": response["output_text"]})

def main():
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config("Chat with PDF using Langchain", layout="wide")
    st.header("Chat with PDF using Google Gemini")

    # Sidebar for PDF upload
    with st.sidebar:
        st.title("Upload PDF")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    # Main chat interface
    chat_container = st.container()

    # Display chat history
    with chat_container:
        for message in st.session_state["chat_history"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # User input
    user_question = st.chat_input("Ask a question about the PDF")
    if user_question:
        user_input(user_question)
        # Rerun the app to update the chat history
        st.rerun()

if __name__ == "__main__":
    main()
