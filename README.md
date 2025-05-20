# SentinelAI

SentinelAI is a Django-based application for intelligent document processing, question answering, and automatic question generation using Retrieval-Augmented Generation (RAG) techniques and LLM integration.

## Features

- **Document Management**: Upload, process, and manage documents for question answering
- **RAG-based Q&A**: Ask questions about your documents and get AI-generated answers with source citations
- **Automatic Question Generation**: Generate relevant questions from your documents automatically
- **Question Categorization**: Organize generated questions by category for better navigation
- **Groq API Integration**: Leverages Groq's large language models for high-quality responses

## Prerequisites

- Python 3.8+
- Django 4.0+
- Groq API key (obtain from https://console.groq.com/keys)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/SentinelAI.git
   cd SentinelAI
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Apply migrations:
   ```
   python manage.py migrate
   ```

5. Run the development server:
   ```
   python manage.py runserver
   ```

6. Access the application at `http://127.0.0.1:8000/`

## Usage

1. **API Key Setup**: 
   - First, set up your Groq API key in the application through the API Key Setup page
   - This is required for document processing and question answering

2. **Upload Documents**: 
   - Navigate to the RAG Upload page
   - Upload PDF, TXT, or other supported document formats
   - Documents will be processed and made available for question answering

3. **Ask Questions**: 
   - Use the Q&A interface to ask questions about your processed documents
   - The system will retrieve relevant content and generate answers with source citations

4. **Generate Questions**: 
   - Upload documents to automatically generate relevant questions
   - The system will analyze the document structure and content to create meaningful questions

5. **View Generated Questions**: 
   - Browse through generated questions organized by category
   - Click on questions to view AI-generated answers

## Project Structure

- `sentinelai_app/`: Main Django application
  - `models.py`: Database models for documents and questions
  - `views.py`: View functions for handling requests
  - `forms.py`: Form definitions for data input
  - `utils.py`: Utility functions for document processing and RAG
  - `templates/`: HTML templates for the user interface

## Technologies

- Django: Web framework
- LangChain: For RAG implementation
- Groq API: LLM provider for generating answers and questions
- FAISS/Chroma: Vector store for document embeddings

## License

[MIT License](LICENSE) 