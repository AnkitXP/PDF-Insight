# PDF Insight: AI-Powered Document Chatbot

PDF Insight is an interactive chatbot application that allows users to upload PDF documents and engage in natural language conversations about their content. Leveraging the power of Streamlit, LangChain, and Google's Gemini AI, this project enables efficient document analysis and information retrieval.

## Features

- **PDF Upload**: Upload multiple PDF documents for analysis
- **Natural Language Processing**: Engage in conversations about document content using natural language
- **AI-Powered Responses**: Utilizes Google's Gemini AI for intelligent question answering
- **Vector Search**: Implements FAISS for efficient similarity search on document content
- **User-Friendly Interface**: Clean and intuitive chat-like interface built with Streamlit

## Installation

To set up PDF Insight on your local machine, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/AnkitXP/pdf-insight.git
    cd pdf-insight
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up your Google API key:
    - Create a `.env` file in the project root
    - Add your Google API key:
      ```
      GOOGLE_API_KEY=your_api_key_here
      ```

## Usage

To run the application:

1. Execute the following command:
    ```bash
    streamlit run app.py
    ```

2. Open your web browser and navigate to the provided local URL (usually `http://localhost:8501`)

3. Upload PDF documents using the sidebar

4. Start chatting with the AI about the content of your uploaded documents

## How It Works

1. **PDF Processing**: The application extracts text from uploaded PDF documents
2. **Text Chunking**: Extracted text is split into manageable chunks
3. **Embedding Generation**: Text chunks are converted into vector embeddings
4. **Similarity Search**: User queries are matched against the most relevant text chunks
5. **AI-Powered Responses**: Google's Gemini AI generates responses based on the relevant context

## Technologies Used

- **Streamlit**: For the web interface
- **LangChain**: For building the conversational AI pipeline
- **Google Generative AI**: For natural language processing and response generation
- **FAISS**: For efficient similarity search
- **PyPDF2**: For PDF text extraction

## Contributing

Contributions to PDF Insight are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
