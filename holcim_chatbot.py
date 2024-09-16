import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.document_loaders import AsyncChromiumLoader
import dotenv
import sys
import logging
import pysqlite3
import streamlit as st
from constant import URL_RETRIEVAL_TEMPLATE, FORMAT_INSTRUCTIONS
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, UnstructuredImageLoader
from langchain_googledrive.retrievers import GoogleDriveRetriever
import warnings
from langchain.tools import tool
from dotenv import load_dotenv
import requests
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from google.oauth2 import service_account
import io

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

logger.info("Logging is configured.")

# Configure SQLite
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

# Load environment variables from a .env file
load_dotenv(dotenv_path='.env')

embedding_function = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2")

llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")


@tool
def google_drive_retriever(input: str) -> str:
    '''
         Args:
            input (str): The search query to locate specific documents within Google Drive.

        Returns:
            str: The content of the retrieved documents.

        This function uses the GoogleDriveRetriever tool to search within a specified folder in Google Drive.
        It supports querying documents of specific types (e.g., PNG images, PDF files) and returns the content
        of the top results based on the search query.
    '''
    tool = GoogleDriveRetriever(
        folder_id="1yr7B5ldBEatJyk71VxlMr6FI7DyBhJzX",
        template="gdrive-query-in-folder",
        conv_mapping={'image/png': UnstructuredImageLoader,
                      'application/pdf': PyPDFLoader},
        mode="documents",
        num_results=5,
    )
    docs = tool.invoke(input)
    return [i.page_content for i in docs]


def setup_retrieval_qa_chain():
    """
    Sets up a retrieval-based question-answering (QA) chain using a Chroma database and a specified prompt template.

    This function initializes a Chroma database with a specified embedding function and collection name, 
    converts it to a retriever, and then creates a RetrievalQA chain using the provided language model (LLM) 
    and retriever. The chain is configured with a specific prompt template and chain type. Finally, it 
    creates a tool named "retrieval_tool" that invokes the QA chain to answer questions based on the provided context.

    Returns:
        Tool: A tool configured to use the retrieval-based QA chain for answering questions.
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(URL_RETRIEVAL_TEMPLATE)

    db3 = Chroma(persist_directory="./my_chroma_data",
                 embedding_function=embedding_function, collection_name='test_collection')
    db_ret = db3.as_retriever()

    chainSim = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db_ret,
        chain_type="stuff",
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    retrieval_tool = Tool(
        name="retrieval_tool",
        func=chainSim.invoke,
        description="If the user does not mention Google Drive, utilize the retrieval_tool to obtain and answer questions based on the provided context."
    )

    return retrieval_tool


def upload_pdf_url_to_google_drive(file_url, folder_id="1yr7B5ldBEatJyk71VxlMr6FI7DyBhJzX"):
    """
    Uploads a PDF file from a given URL to a specified folder in Google Drive.

    Args:
        file_url (str): The URL of the PDF file to be uploaded.
        folder_id (str, optional): The ID of the Google Drive folder where the file will be uploaded. 
                                   Defaults to "1yr7B5ldBEatJyk71VxlMr6FI7DyBhJzX".

    Returns:
        str: The ID of the uploaded file in Google Drive.

    Raises:
        Exception: If the file download from the URL fails.
    """
    # Load credentials from the service account file
    credentials = service_account.Credentials.from_service_account_file(
        os.environ.get('GOOGLE_ACCOUNT_FILE'),
        scopes=['https://www.googleapis.com/auth/drive']
    )

    # Initialize the Google Drive API client
    service = build('drive', 'v3', credentials=credentials)

    # Download the PDF file from the URL
    response = requests.get(file_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download file: {response.status_code}")

    # Create a MediaIoBaseUpload object
    file_stream = io.BytesIO(response.content)
    media = MediaIoBaseUpload(file_stream, mimetype='application/pdf')

    # Create a file metadata object
    file_metadata = {
        'name': file_url.split("/")[-1],  # Use the file name from the URL
        'parents': [folder_id]
    }

    # Upload the file to Google Drive
    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()

    return file.get('id')


def setup_agent(retrieval_tool):
    """
    Initializes and sets up an agent with the provided retrieval tool.

    Args:
        retrieval_tool: The retrieval tool to be used by the agent.

    Returns:
        The initialized agent configured with the specified tools and settings.
    """
    agent = initialize_agent(
        tools=[google_drive_retriever, retrieval_tool],
        llm=llm,
        verbose=True,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent_kwargs={
            'format_instructions': FORMAT_INSTRUCTIONS,
        },
        handle_parsing_errors=True
    )

    return agent


def scrape_with_playwright(urls):
    """
    Scrapes web pages using Playwright, transforms the HTML content to text, and splits the text into chunks.

    Args:
        urls (list of str): A list of URLs to scrape.

    Returns:
        list: A list of text chunks obtained from the scraped web pages.
    """
    loader = AsyncChromiumLoader(urls)
    docs = loader.load()

    transformer = Html2TextTransformer()
    text_content = transformer.transform_documents(docs)

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    splits = splitter.split_documents(text_content)
    return splits


def persist_data_in_chroma(extracted_content, embedding_function):
    """
    Persists extracted content into a Chroma vector database.

    Args:
        extracted_content (list): A list of documents to be embedded and stored.
        embedding_function (callable): A function that generates embeddings for the documents.

    Returns:
        Chroma: An instance of the Chroma vector database containing the persisted data.
    """
    vectordb = Chroma.from_documents(
        collection_name="test_collection",
        documents=extracted_content,
        embedding=embedding_function,
        persist_directory="./my_chroma_data"
    )
    return vectordb


def main():
    """
    Main function to run the Holcim chatbot application.

    This function sets up the Streamlit interface for the Holcim chatbot, allowing users to:
    - Ask questions and get answers using a retrieval-based QA chain.
    - Scrape content from provided URLs and persist the data.
    - Upload files to Google Drive either from a URL or from the local system.

    The interface includes:
    - A text input for entering questions.
    - A button to get answers to the entered questions.
    - A text area for entering URLs to scrape.
    - A button to initiate the scraping of the entered URLs.
    - A subheader and radio buttons for choosing the method to upload files to Google Drive.
    - A text input for entering the file URL (if the upload method is "Upload from URL").
    - A button to upload the file from the entered URL.

    The function handles the following actions:
    - Setting up the retrieval QA chain and agent to answer questions.
    - Scraping URLs using Playwright and persisting the data.
    - Uploading files to Google Drive from a URL.

    Note: The function assumes the existence of helper functions such as `setup_retrieval_qa_chain`, 
    `setup_agent`, `scrape_with_playwright`, `persist_data_in_chroma`, and `upload_pdf_url_to_google_drive`.
    """
    st.title("Chat with Holcim")
    question = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        retrieval_tool = setup_retrieval_qa_chain()
        agent = setup_agent(retrieval_tool)
        answer = agent.run(question)
        st.write(answer)

    urls = st.text_area("Enter URLs to scrape (comma-separated):")

    if st.button("Scrape URLs"):
        urls_list = [url.strip() for url in urls.split(",")]
        extracted_content = scrape_with_playwright(urls_list)
        persist_data_in_chroma(
            extracted_content=extracted_content, embedding_function=embedding_function)
        st.success(
            "Scraping and data persistence completed. Now you can query to Get Answer from the URLs.")

    st.subheader("Upload Files to Google Drive")
    upload_option = st.radio("Choose upload method:",
                             ("Upload from URL", "Upload from Local"))

    if upload_option == "Upload from URL":
        file_url = st.text_input("Enter the file URL:")
        if st.button("Upload from URL"):
            if file_url:
                file_id = upload_pdf_url_to_google_drive(file_url)
                st.success(f"File {file_id} uploaded successfully from URL.")
            else:
                st.error("Please enter a valid URL.")


if __name__ == "__main__":
    main()
