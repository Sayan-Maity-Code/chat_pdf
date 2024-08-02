import os# It is used to interacting with our os , we used it to fetch environment variables
import shutil# is used to handle file operations related to the FAISS index directory. Specifically, it ensures that any existing FAISS index directory is deleted before creating a new one.
import gradio as gr# is for the ui
from dotenv import load_dotenv#to load env_variables from .env file
from llama_parse import LlamaParse#(is a document parsing tool) It allows the system to process and analyze document content, which is essential for further operations like indexing, summarization, and question-answering.
from langchain_community.vectorstores import FAISS#(Facebook AI Similarity Search),(for efficient similarity search and clustering of dense vectors),It enables the system to quickly find and retrieve relevant document sections based on query similarity, which is crucial for effective information retrieval.
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline#Embeddings capture semantic meaning, allowing the system to compare and search texts based on their content, rather than just keyword matching,Pipeline provides a simple interface to various NLP tasks using pre-trained models.
from langchain.chains import RetrievalQA# It integrates retrieval mechanisms and language models to provide accurate and contextually relevant answers
from langchain.prompts import PromptTemplate# helps the model understand the input structure and respond appropriately.
from langchain.retrievers import ContextualCompressionRetriever#It retrieves and compresses document content, making it suitable for models with input length limitations.
from langchain.retrievers.document_compressors import LLMChainExtractor#It extracts essential parts of the document to help answer specific queries
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline#AutoTokenizer is used to tokenize the input text, breaking it down into tokens that the model can understand.
#AutoModelForSeq2SeqLM is a pre-trained sequence-to-sequence language model used for generating text based on the input tokens.
#it's used to create a text generation pipeline for generating answers or summaries.
from langchain.text_splitter import RecursiveCharacterTextSplitter#This tool splits long texts into smaller, manageable chunks.

# Load environment variables
load_dotenv()


# Initialize LlamaParse
parser_key = os.getenv("llama_parser")  # Fetches the LlamaParse API key from environment variables.
#It only parse 1000 pages per day for free or you will have to upgrade the plan 🤑💸
'''https://cloud.llamaindex.ai/api-key'''#and there must be 1200 pages per file max
parser = LlamaParse(api_key=parser_key, result_type="markdown", num_workers=4, verbose=True)  # Initializes LlamaParse with the API key and specific settings.
# Using markdown makes the output easier to read and understand.
#num_workers=4, the parsing task can be distributed across four workers, allowing for parallel processing.
#When verbose is set to True, the parser provides detailed logging information about the parsing process.



# Global variables
loaded_db = None  # Will hold the loaded FAISS database.
qa_chain = None  # Will hold the RetrievalQA chain.
document_summary = ""  # Will hold the document summary.
tokenizer = None  # Will hold the tokenizer for the language model.
model = None  # Will hold the language model.

def extract_documents(file):
    return parser.load_data(file.name)## Uses LlamaParse to extract text from the uploaded file.

def process_document(file):
    global loaded_db, qa_chain, document_summary, tokenizer, model# Declare global variables to modify them from anywhere.

    documents = extract_documents(file)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,  # Maximum size of each text chunk.
        chunk_overlap=20,  # Overlap between chunks to maintain context.
        length_function=len,  # Function to determine chunk length.
    )

    texts = []  # List to store text chunks.
    metadatas = []  # List to store metadata for each chunk.
    for doc in documents:  # Loop through each extracted document.
        chunks = text_splitter.split_text(doc.text)  # Split document text into chunks.
        texts.extend(chunks)  # Add chunks to the texts list.
        metadatas.extend([doc.metadata] * len(chunks))  # Add metadata for each chunk.

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Initialize opensource embeddings model.

    if os.path.exists("faiss_index"):  # Check if FAISS index directory exists.
        shutil.rmtree("faiss_index")  # Remove existing FAISS index directory.
    db = FAISS.from_texts(texts, embeddings, metadatas=metadatas)  # Create FAISS index from texts and embeddings.
    db.save_local("faiss_index")  # Save the FAISS index locally.
    loaded_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)  # Load the saved FAISS index.It tells the FAISS library to skip some of the safety checks that are typically enforced during deserialization.


    model_name = "google/flan-t5-large"  # Specify the language model to use.
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # Initialize the tokenizer for the language model.
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)  # Initialize the language model.

    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)  # Create a text generation pipeline.
    llm = HuggingFacePipeline(pipeline=pipe)  # Initialize HuggingFacePipeline with the text generation pipeline.

    prompt_template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Provide a detailed explanation in your answer, using up to five sentences.

    Context: {context}

    Question: {question}
    Detailed Answer:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])# Initialize the prompt template.

    compressor = LLMChainExtractor.from_llm(llm)  # Initialize the document compressor.
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,  # Set the document compressor.
        base_retriever=loaded_db.as_retriever(search_kwargs={"k": 4})  #loaded_db.as_retriever(): This method converts the FAISS index (loaded_db) into a retriever object that can be used to perform similarity searches.{"k": 4} indicates that the retriever should return the top 4 most similar documents or text chunks based on the similarity score.
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,  # Set the language model.
        chain_type="stuff",  #Choosing the Right Chain Type
        #"stuff": Simple and direct, good for straightforward contexts.
        #"map_reduce": Best for complex or large documents requiring summarization.
        #"refine": Suitable for detailed or nuanced answers requiring refinement.
        #"qa": Optimized for direct question-answering tasks.
        retriever=compression_retriever,  #retriever=compression_retriever: Configures the QA chain to use a retriever that not only fetches relevant documents but also compresses them to ensure the input to the language model is concise and effective.
        return_source_documents=True,  # Indicate to return source documents.
        chain_type_kwargs={"prompt": PROMPT}  # Set additional chain type parameters.
    )

    document_summary = summarize_document(loaded_db)  # Generate a summary of the document.

    return "Document processed successfully. You can now ask questions or view the summary."  # Return a success message.

def summarize_document(db, chunk_size=400):
    all_docs = db.similarity_search("", k=db.index.ntotal)  # Retrieve all documents from the database.
    full_text = " ".join(doc.page_content for doc in all_docs)  # Combine all document texts.
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]  # Split the full text into chunks.
    return " ".join(chunks)  # Return the combined chunks as the summary.

def answer_question(query):
    if not query.strip():  # Check if the query is empty.
        return "Please enter a question before clicking the Ask button."
    
    if not loaded_db or not qa_chain:  # Check if the database or QA chain is not initialized.
        return "Please process a document first."

    result = qa_chain.invoke({"query": query})  # Get the answer from the QA chain.
    return result["result"].strip().rstrip("[1]").strip()  # Clean up and return the answer.


# Gradio interface
with gr.Blocks(".gradio-container {background: url(file='./img.jpeg')}") as demo:  # Create a Gradio interface with custom CSS.
    gr.Markdown("# Intelligent Document Q&A System")  # Add a title.

    with gr.Row():
        file_input = gr.File(label="Upload Document")  # Add a file upload input.
        process_button = gr.Button("Process Document", elem_classes=["gradio-button"])  # Add a button to process the document.

    status_output = gr.Textbox(label="Status", elem_classes=["gradio-textbox"])  # Add a textbox for status messages.

    with gr.Row():
        question_input = gr.Textbox(label="Ask a question", elem_classes=["gradio-textbox"])  # Add a textbox for inputting questions.
        answer_output = gr.Textbox(label="Answer", elem_classes=["gradio-textbox"])  # Add a textbox for displaying answers.

    ask_button = gr.Button("Ask", elem_classes=["gradio-button"])  # Add a button to ask a question.

    summary_output = gr.Textbox(label="Document Summary", lines=10, elem_classes=["gradio-textbox"])  # Add a textbox for the document summary.
    summary_button = gr.Button("Generate Summary", elem_classes=["gradio-button"])  # Add a button to generate the summary.

    process_button.click(process_document, inputs=[file_input], outputs=[status_output])  # Define the click action for the process button.
    ask_button.click(answer_question, inputs=[question_input], outputs=[answer_output])  # Define the click action for the ask button.
    summary_button.click(lambda: document_summary, outputs=[summary_output])  # Define the click action for the summary button.

# Launch the interface
demo.launch(share=True)  # Launch the Gradio interface with sharing enabled.



