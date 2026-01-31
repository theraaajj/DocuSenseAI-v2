import tempfile
import os
import pandas as pd
import ollama  
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_uploaded_file(uploaded_file):
    """
    Ingests PDF, DOCX, XLSX, CSV, TXT, or MD.
    Handles multiple Excel sheets and merged cells better, and returns the vector store.
    """
    # save to temp file (preserving extension -crucial)
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        loader = None
        raw_docs = []

        # pdf
        if file_ext == ".pdf":
            loader = PyPDFLoader(tmp_path)
            raw_docs = loader.load()

        # excel with both extensions
        elif file_ext in [".xlsx", ".xls"]:
            xls = pd.read_excel(tmp_path, sheet_name=None)
            full_text = []
            
            for sheet_name, df in xls.items():
                # force headers to string to avoid type errors
                df.columns = df.columns.astype(str)
                
                # explicit schema extraction
                columns_list = ", ".join(list(df.columns))
                
                # clean NaN
                df = df.fillna("")
                
                # data card for the LLM
                sheet_content = f"""
                --- SHEET: {sheet_name} ---
                COLUMN HEADERS: [{columns_list}]
                
                FIRST 20 ROWS OF DATA:
                {df.head(20).to_markdown(index=False)}
                
                FULL DATA (Markdown):
                {df.to_markdown(index=False)}
                """
                full_text.append(sheet_content)
            
            text_content = "\n".join(full_text)
            raw_docs = [Document(page_content=text_content, metadata={"source": uploaded_file.name})]
        
        # CSV
        elif file_ext == ".csv":
            df = pd.read_csv(tmp_path)
            
            # Explicit Schema (as it threw error in finding column headers)
            df.columns = df.columns.astype(str)
            columns_list = ", ".join(list(df.columns))
            
            # Data Card
            df = df.fillna("")
            text_content = f"""
            FILE: {uploaded_file.name}
            COLUMN HEADERS: [{columns_list}]
            
            FIRST 20 ROWS SAMPLE:
            {df.head(20).to_markdown(index=False)}
            
            FULL DATA:
            {df.to_markdown(index=False)}
            """
            raw_docs = [Document(page_content=text_content, metadata={"source": uploaded_file.name})]

        # TXT, MD, PY
        elif file_ext in [".txt", ".md", ".py"]:
            loader = TextLoader(tmp_path)
            raw_docs = loader.load()

        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

        
        # Final Splitting & Vectorizing
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, 
            chunk_overlap=150,  ## need to play with the chunk overlap during evaluation
            add_start_index=True
        )
        chunks = text_splitter.split_documents(raw_docs)
        
        vector_store = FAISS.from_documents(chunks, embedding_model)
        return vector_store, len(chunks)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def query_local_model(query, vector_store):
    """
    1. Search the vector store for relevant chunks.
    2. Send them to Ollama (Llama 3) to generate a formatted answer.
    """
    # retrieves the top 3 most relevant chunks (need to play with the k during evaluation)
    docs = vector_store.similarity_search(query, k=3)
    
    # context string
    context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])
    
    # System Prompt needs to be strict. will prevent hallucinations
    system_prompt = f"""
    You are DocuSenseAI, a secure local reasoning assistant.
    STRICT RULES:
    1. USE ONLY the provided context.
    2. If the answer is NOT in the context, strictly state: "I cannot find this information in the provided files."
    3. Do NOT invent facts. Do NOT use outside knowledge (like "As an AI...").
    4. If the user asks to "Summarize", provide a structured summary with bullet points.
    5. If the user asks to "Write content", provide the raw text verbatim.

    CONTEXT FROM FILES:
    {context_text}
    """

    # Ollama call
    response = ollama.chat(model='phi3', messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': query},
    ])

    # returns the generated answer, alongwith the source docs
    return response['message']['content'], docs


def extract_search_keyword(user_query):
    """
    Uses Llama 3 to turn a complex sentence into a simple filename keyword.
    Example: "Show me the notes about Solar" -> "Solar"
    """
    system_prompt = """
    You are a Search Query Extractor.
    Extract the single most likely FILENAME keyword or TOPIC from the user's request.
    
    Rules:
    - Return ONLY the keyword.
    - No explanations.
    - If the user asks "Show me the budget", return "budget".
    - If the user asks "Read the file named data.csv", return "data".
    
    User Query:
    """
    
    try:
        response = ollama.chat(model='llama3', messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_query},
        ])
        # extra whitespace or punctuation the model added, has to be removed
        keyword = response['message']['content'].strip().replace('"', '').replace("'", "")
        return keyword
    except Exception as e:
        return user_query # fallback