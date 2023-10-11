# Insight
Insight Application Codebase
```markdown
# Document Summarization and CSV Chat App

This Streamlit web application is designed to help users perform document summarization and chat with CSV data using the Language Model GPT-3.5, developed by OpenAI. It allows users to upload a PDF document for summarization and chat with the content of a CSV file.

## Table of Contents

- [Getting Started](#getting-started)
- [Usage](#usage)
- [Features](#features)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [File Structure](#file-structure)
- [Acknowledgments](#acknowledgments)

## Getting Started

To run this application, you will need to install the necessary dependencies. We recommend using a virtual environment to manage your dependencies. Follow these steps:

1. Clone this repository to your local machine:

   ```
   git clone https://github.com/your-repo-url.git
   ```

2. Change into the project directory:

   ```
   cd document-summarization-and-csv-chat
   ```

3. Install the required dependencies using pip:

   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file and add your OpenAI API key as follows:

   ```
   OPENAI_API_KEY=your-api-key
   ```

5. Run the Streamlit app:

   ```
   streamlit run main.py
   ```

The application should now be accessible in your web browser.

## Usage

### Document Summarization

1. Upload a PDF file that you want to summarize.
2. Enter your question or query in the provided input box.
3. Click the "Summarize" button.
4. The application will summarize the document based on your query and display the results.

### Chat with CSV

1. Upload a CSV file containing your data.
2. Enter your query in the text area.
3. Click the "Chat with CSV" button.
4. The application will use GPT-3.5 to generate responses based on the data in the CSV file and your query.

## Features

- Document summarization using DistilBERT model.
- CSV data chat using GPT-3.5 (Language Model by OpenAI).
- Upload PDF and CSV files for processing.
- Interactive and user-friendly web interface.
- Seamless integration with the OpenAI API.

## Dependencies

- Streamlit
- PyTorch
- Transformers
- NLTK
- PyPDF2
- Pandas
- Dotenv

You can install the required dependencies using the `requirements.txt` file.

## Configuration

The application requires an OpenAI API key, which should be placed in a `.env` file in the project directory. Create a `.env` file and add your API key as shown in the "Getting Started" section.

## File Structure

- `main.py`: The main Streamlit application.
- `readme.md`: This documentation.
- `requirements.txt`: List of required Python packages.
- `.env.example`: Example environment file (for OpenAI API key).
- `data/`: Directory for storing uploaded PDF and CSV files.

## Acknowledgments

This application utilizes the power of OpenAI's GPT-3.5 and Hugging Face's DistilBERT for document summarization. Special thanks to the open-source community for their contributions to the libraries used in this project.
```

Remember to replace `your-repo-url` with the actual URL of your repository and customize the content to match your project's specifics. This `readme.md` provides a basic structure and information that will help users understand and use your application.
