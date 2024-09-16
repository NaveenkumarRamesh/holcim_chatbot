# Holcim Chatbot

This repository contains the code for the Holcim Chatbot, a Streamlit-based application that allows users to interact with a retrieval-based question-answering (QA) system. The chatbot can answer questions based on provided context, scrape content from URLs, and upload files to Google Drive.

## Features

- **Question Answering**: Ask questions and get answers using a retrieval-based QA chain and from Google Drive Files such as PDF/Jpeg/text/png,.
- **Web Scraping**: Scrape content from provided URLs and persist the data in a Chroma vector database.
- **File Upload**: Upload files to Google Drive either from a URL or from the local system.

## Requirements

- Python 3.7+
- Streamlit
- LangChain
- SentenceTransformers
- Chroma
- Google API Client
- Requests
- PyMuPDF
- pdfplumber
- dotenv

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/holcim_chatbot.git
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up environment variables:
    - Create a `.env` file in the root directory.
    - Add the following environment variables:
      ```
      GOOGLE_DRIVE_API_KEY=your_api_key
      GOOGLE_ACCOUNT_FILE=credentials_to_gdrive
      ```

## Usage

1. Run the Streamlit app:
    ```bash
    streamlit run holcim_chatbot.py
    ```

2. Interact with the Chatbot:
   - **Ask Questions**: Enter your question in the text input and click "Get Answer".
   - **Scrape URLs**: Enter URLs (comma-separated) in the text area and click "Scrape URLs".
   - **Upload Files to Google Drive**: Choose the upload method, enter the file URL or select a local file, and click the corresponding upload button.

## File Structure

- `holcim_chatbot.py`: Main script containing the Streamlit app and chatbot logic.
- `constant.py`: Contains constants and templates used in the application.
- `qauth.py`: Handles Google Drive authentication.
- `requirements.txt`: Lists the required Python packages.

## Functions

### `main()`
The main function to run the Holcim chatbot application. Sets up the Streamlit interface and handles user interactions.

### `setup_retrieval_qa_chain()`
Sets up a retrieval-based QA chain using a Chroma database and a specified prompt template.

### `setup_agent(retrieval_tool)`
Initializes and sets up an agent with the provided retrieval tool.

### `scrape_with_playwright(urls)`
Scrapes web pages using Playwright, transforms the HTML content to text, and splits the text into chunks.

### `persist_data_in_chroma(extracted_content, embedding_function)`
Persists extracted content into a Chroma vector database.

### `upload_pdf_url_to_google_drive(file_url, folder_id)`
Uploads a PDF file from a given URL to a specified folder in Google Drive.

### `google_drive_retriever(input)`
Retrieves documents from Google Drive based on the search query.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- Streamlit
- LangChain
- SentenceTransformers
- Chroma
- Google API Client

## Contact

For any questions or issues, please open an issue on the GitHub repository or contact the maintainer at your.email@example.com.
