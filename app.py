import os
import shutil
import gradio as gr
from dotenv import load_dotenv
from llama_parse import LlamaParse
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Initialize LlamaParse
parser_key = os.getenv("llama_parser")
parser = LlamaParse(api_key=parser_key, result_type="markdown", num_workers=4, verbose=True)

# Global variables
loaded_db = None
qa_chain = None
document_summary = ""
tokenizer = None
model = None

def extract_documents(file):
    return parser.load_data(file.name)

def process_document(file):
    global loaded_db, qa_chain, document_summary, tokenizer, model

    documents = extract_documents(file)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        length_function=len,
    )

    texts = []
    metadatas = []
    for doc in documents:
        chunks = text_splitter.split_text(doc.text)
        texts.extend(chunks)
        metadatas.extend([doc.metadata] * len(chunks))

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")

    db = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    db.save_local("faiss_index")
    loaded_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
    llm = HuggingFacePipeline(pipeline=pipe)

    prompt_template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Provide a detailed explanation in your answer, using up to five sentences.

    Context: {context}

    Question: {question}
    Detailed Answer:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=loaded_db.as_retriever(search_kwargs={"k": 4})
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=compression_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    document_summary = summarize_document(loaded_db)

    return "Document processed successfully. You can now ask questions or view the summary."

def summarize_document(db, chunk_size=400):
    all_docs = db.similarity_search("", k=db.index.ntotal)
    full_text = " ".join(doc.page_content for doc in all_docs)
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
    return " ".join(chunks)

def answer_question(query):
    if not query.strip():
        return "Please enter a question before clicking the Ask button."
    
    if not loaded_db or not qa_chain:
        return "Please process a document first."

    result = qa_chain.invoke({"query": query})
    return result["result"].strip().rstrip("[1]").strip()

# Gradio interface
with gr.Blocks(".gradio-container {background: url(file='./img.jpeg')}") as demo:
    gr.Markdown("# Intelligent Document Q&A System")
    
    with gr.Row():
        file_input = gr.File(label="Upload Document")
        process_button = gr.Button("Process Document", elem_classes=["gradio-button"])
    
    status_output = gr.Textbox(label="Status", elem_classes=["gradio-textbox"])
    
    with gr.Row():
        question_input = gr.Textbox(label="Ask a question", elem_classes=["gradio-textbox"])
        answer_output = gr.Textbox(label="Answer", elem_classes=["gradio-textbox"])
    
    ask_button = gr.Button("Ask", elem_classes=["gradio-button"])
    
    summary_output = gr.Textbox(label="Document Summary", lines=10, elem_classes=["gradio-textbox"])
    summary_button = gr.Button("Generate Summary", elem_classes=["gradio-button"])
    
    process_button.click(process_document, inputs=[file_input], outputs=[status_output])
    ask_button.click(answer_question, inputs=[question_input], outputs=[answer_output])
    summary_button.click(lambda: document_summary, outputs=[summary_output])

# Launch the interface
demo.launch(share=True)



