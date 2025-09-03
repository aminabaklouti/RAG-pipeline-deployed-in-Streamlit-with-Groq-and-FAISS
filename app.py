import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
## create_retrieval_chain → ties retrieval + LLM into one pipeline.
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
## FAISS → vector store for similarity search.
import time
load_dotenv()

groq_api_key=os.environ['GROQ_API_KEY']

## Streamlit session state ensures that embeddings, documents, and FAISS vector store 
# are only computed once (not on every page reload).

if "vector" not in st.session_state:
    st.session_state.embeddings=OllamaEmbeddings(model="llama3.2:latest")
    #WebBaseLoader(...) → fetches text from LangSmith docs.
    st.session_state.loader=WebBaseLoader("https://docs.langchain.com/langsmith/home")
    #.load() → loads the actual text into memory.
    st.session_state.docs=st.session_state.loader.load()
    #RecursiveCharacterTextSplitter(...) → splits docs into ~1000-character chunks with 200 overlap.
    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
    #builds a FAISS index for similarity search.
    st.session_state.vector=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

st.title("RAG pipeline deployed in Streamlit with Groq and FAISS")
llm=ChatGroq(api_key=groq_api_key,model="llama-3.3-70b-versatile")

## Prompt Template
# context → retrieved document chunks.
# input → user’s query.
prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions:{input}
    """ 
)

## Build the Retrieval-Augmented Pipeline
document_chain=create_stuff_documents_chain(llm,prompt)
# create_stuff_documents_chain(...) → creates a chain that combines multiple docs into one context for the LLM.
retriever=st.session_state.vector.as_retriever()
# retriever=... → FAISS retriever to fetch relevant chunks.
retrieval_chain=create_retrieval_chain(retriever, document_chain)
# create_retrieval_chain(...) → RAG pipeline: retrieval → document_chain → answer.
prompt=st.text_input("Input your prompt here")

if prompt:
    start=time.process_time()
    response=retrieval_chain.invoke({"input":prompt})
    # runs retrieval + LLM inference.
    print("Response time: ",time.process_time()-start)
    st.write(response['answer'])

    # With a streamlit expander
    # Show Retrieved Context
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
