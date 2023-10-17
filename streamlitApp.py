import streamlit as st
import textwrap
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings

# Set up the Streamlit app
st.title('Question Answering App')

# Sidebar contents
with st.sidebar:
    st.markdown('## About')
    st.write('Langchain Document QA')
    st.write('Made by BrigBox')

# Load and process the text files
# loader = TextLoader('single_text_file.txt')
loader = DirectoryLoader('./new_papers/new_papers/', glob="./*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# Splitting the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Initialize instructor embeddings
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs={"device": "cuda"})

# Embed and store the texts
persist_directory = 'db'
embedding = instructor_embeddings
vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)
vectordb.persist()

# Create the retriever
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
retriever = vectordb.as_retriever()
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# Create the chain to answer questions
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)

# Function to wrap text while preserving newlines
def wrap_text_preserve_newlines(text, width=110):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

# Function to process LLM response and display it
def process_llm_response(llm_response):
    st.write(wrap_text_preserve_newlines(llm_response['result']))
    st.write('Sources:')
    for source in llm_response["source_documents"]:
        st.write(source.metadata['source'])

# User input and question-answering
query = st.text_input("Ask a question about the documents")
if query:
    llm_response = qa_chain(query)
    process_llm_response(llm_response)

# Cleanup option
if st.button("Cleanup and Delete Collection"):
    vectordb.delete_collection()
    vectordb.persist()
    st.write("Collection deleted and persisted.")

# Delete the directory (optional)
# !rm -rf db/
