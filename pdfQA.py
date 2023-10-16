from transformers import DPRContextEncoder, DPRQuestionEncoder, BertTokenizer
import torch
import PyPDF2
import re

# Define paths to the pretrained DPR models and tokenizers
context_encoder_path = 'facebook/dpr-ctx_encoder-multiset-base'
question_encoder_path = 'facebook/dpr-question_encoder-multiset-base'

# Initialize the DPR question and context encoders
question_encoder = DPRQuestionEncoder.from_pretrained(question_encoder_path)
context_encoder = DPRContextEncoder.from_pretrained(context_encoder_path)


# Function to split the text into sentences using a simple regex
def extract_sentences(text):
    # This regex splits text into sentences based on common punctuation marks
    sentence_delimiters = re.compile(r'[.!?]')
    sentences = sentence_delimiters.split(text)
    return sentences

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    pdf_text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
    return pdf_text

# User's question
user_question = input("Enter your question: ")

# Tokenize and encode the user's question
# Initialize a BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
encoded_question = tokenizer.encode_plus(
    user_question,
    add_special_tokens=True,
    return_tensors="pt",
    padding="max_length",
    max_length=64,  # Adjust as needed
    truncation=True,
)

# Encode the question using the DPR question encoder
question_embedding = question_encoder(**encoded_question)["pooler_output"]

# Example corpus of text documents (replace with your actual corpus)
corpus = [extract_text_from_pdf("docFull.pdf"), extract_text_from_pdf("SOPTSS.pdf")]

# Encode each document in the corpus and calculate cosine similarity
retrieved_documents = []
for doc in corpus:
    sentences = extract_sentences(doc)  # Extract sentences from the document

    for sentence in sentences:
        # Tokenize and encode the sentence
        encoded_doc = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            max_length=64,  # Adjust as needed
            truncation=True,
        )

        # Encode the sentence using the DPR context encoder
        doc_embedding = context_encoder(**encoded_doc)["pooler_output"]

        # Calculate cosine similarity between the question and sentence embeddings
        similarity = torch.nn.functional.cosine_similarity(question_embedding, doc_embedding, dim=1)

        retrieved_documents.append((sentence, similarity.item()))

# Sort documents by similarity (higher similarity means more relevant)
retrieved_documents = sorted(retrieved_documents, key=lambda x: x[1], reverse=True)

# Get the most relevant sentences
top_sentences = [doc for doc, _ in retrieved_documents[:3]]

# Combine the sentences into one text
document = "\n".join(top_sentences)

# Now you have the relevant document to perform question-answering on
print(document)