import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from PyPDF2 import PdfReader
import os
from keybert import KeyBERT
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# MongoDB setup
MONGODB_URI = "mongodb+srv://prakashkofficials:prakash%402812@cluster0.7yuit.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(MONGODB_URI)
db = client['document_summarization_db']
collection = db['summaries']

# Set the page configuration at the beginning
st.set_page_config(page_title="Document Summarization", layout="wide")

st.header("Document Summarization and Keyword Extraction")

# Getting Groq API Key
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

# Checking for Groq API Key
if not groq_api_key:
    st.warning("Please provide a valid Groq API Key.")

# Getting PDF files
pdf_docs = st.sidebar.file_uploader("Upload your PDF Files and Click on the Submit Button", accept_multiple_files=True)

if st.sidebar.button("Submit"):

    if not pdf_docs:
        st.error("Please upload at least one PDF file.")
    else:
        # Text extraction from PDF
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""  # Ensuring empty strings are handled

        if not text.strip():
            st.error("No text could be extracted from the uploaded PDF(s). Please try again.")
        else:
            # Converting entire text into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
            docs = splitter.create_documents([text])

            # Creating chain for summarization
            llm_model = ChatGroq(model="llama3-8b-8192", groq_api_key=groq_api_key)
            chain = load_summarize_chain(llm=llm_model, chain_type="refine", verbose=True)
            output_summary = chain.run(docs)

            # Extracting keywords
            keyword_extraction_model = KeyBERT()
            keywords_with_scores = keyword_extraction_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 1),
                stop_words="english",
                use_maxsum=True,
                diversity=0.2,
                top_n=20
            )
            keyword_extracted = [keyword for keyword, score in keywords_with_scores]

            # Printing summarized text
            st.write("### Summary:")
            st.success(output_summary)

            st.write("### Keywords:")
            if keyword_extracted:
                for keyword in keyword_extracted:
                    st.success(keyword)  # Display each keyword separately
            else:
                st.warning("No keywords extracted from the text.")

                # Prepare the document data for insertion into MongoDB
            document_data = {
                    "summary": output_summary,
                    "keywords": keyword_extracted,
                }

                # Insert data into MongoDB
            try:
                    result = collection.insert_one(document_data)
                    st.success(f"Data inserted into MongoDB with document ID: {result.inserted_id}")
            except Exception as e:
                    st.error(f"An error occurred while inserting data into MongoDB: {str(e)}")

# Close MongoDB connection (optional, handled automatically by garbage collection)
client.close()
