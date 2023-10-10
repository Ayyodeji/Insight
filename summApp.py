import streamlit as st
import torch
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    pipeline,
)
import spacy
import PyPDF2
import io

# Initialize spaCy model for sentence tokenization
nlp = spacy.load("en_core_web_sm")

# Initialize models and tokenizers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased').to(device)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Function to tokenize a document into an array of sentences using spaCy
def document_to_sentences(document):
    sentences = [sent.text for sent in nlp(document).sents]
    return sentences

# Function to extract text from uploaded PDFs
def extract_text_from_uploaded_pdf_using_reader(uploaded_files):
    try:
        pdf_text = ''

        if isinstance(uploaded_files, list):
            # Handle multi-file upload (list of UploadedFile objects)
            for uploaded_file in uploaded_files:
                if uploaded_file.type == "application/pdf":
                    file_like = io.BytesIO(uploaded_file.read())
                    pdf_reader = PyPDF2.PdfReader(file_like)
                    for page in pdf_reader.pages:
                        pdf_text += page.extract_text()
                else:
                    print(f"Skipping {uploaded_file.name} (not a PDF)")
        else:
            # Handle single-file upload (UploadedFile object)
            if uploaded_files.type == "application/pdf":
                file_like = io.BytesIO(uploaded_files.read())
                pdf_reader = PyPDF2.PdfReader(file_like)
                for page in pdf_reader.pages:
                    pdf_text += page.extract_text()
            else:
                print(f"Skipping {uploaded_files.name} (not a PDF)")

        return document_to_sentences(pdf_text)
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return []



# Streamlit app
def main():
    st.title("Document Summarization App")
    
    # Upload PDF file
    uploaded_files = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_files:
        st.subheader("Enter your question:")
        user_question = st.text_input("")

        if st.button("Summarize"):
            st.info("Summarizing...")

            # Extract text from uploaded PDF
            corpus = extract_text_from_uploaded_pdf_using_reader(uploaded_files)

            # Tokenize and encode the user's question
            encoded_question = tokenizer.encode_plus(
                user_question,
                add_special_tokens=True,
                return_tensors="pt",
                padding="max_length",
                max_length=64,  # Adjust as needed
                truncation=True,
            )

            # Encode the question using the DistilBERT model
            question_embedding = model(**encoded_question).last_hidden_state.mean(dim=1)

            # Encode each document in the corpus and calculate cosine similarity
            retrieved_documents = []
            for doc in corpus:
                # Tokenize and encode the document
                encoded_doc = tokenizer.encode_plus(
                    doc,
                    add_special_tokens=True,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=64,  # Adjust as needed
                    truncation=True,
                )

                # Encode the document using the DistilBERT model
                doc_embedding = model(**encoded_doc).last_hidden_state.mean(dim=1)

                # Calculate cosine similarity between the question and document embeddings
                similarity = torch.nn.functional.cosine_similarity(question_embedding, doc_embedding, dim=1)

                retrieved_documents.append((doc, similarity.item()))

            # Sort sentences by similarity (higher similarity means more relevant)
            retrieved_sentences = sorted(retrieved_documents, key=lambda x: x[1], reverse=True)

            # Get the most relevant sentences
            top_sentences = [sent for sent, _ in retrieved_sentences[:2]]  # Summarize fewer sentences

            # Combine the sentences into one document
            document = "\n".join(top_sentences)

            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            summary = summarizer(document, max_length=38, min_length=10, do_sample=True)[0]['summary_text']

            # Display the summarized text
            st.subheader("Summary:")
            st.write(summary)

if __name__ == "__main__":
    main()
