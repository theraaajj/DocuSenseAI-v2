import streamlit as st
from processor import process_uploaded_file, query_local_model
from disk_ops import DiskScout

st.set_page_config(page_title="DocuSenseAI v2.0", layout="wide")

# initialize session state initialization
if "disk_scout" not in st.session_state:
    st.session_state.disk_scout = DiskScout()
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# st.title("🧠 DocuSenseAI v2")

# sidebar for memory control and uploads, access!! 
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=50)
    st.title("DocuSenseAI")
    st.caption("v2.0 | Local | Privacy-First")
    st.divider()

    st.caption("Memory Control")
    # Clear Memory Button
    if st.button("🗑️ Forget All Data"):
        st.session_state.vector_store = None
        st.session_state.disk_scout = DiskScout() # Re-init to clear paths
        st.rerun() # refreshes the app

    # st.header("Data Sources")
    st.divider()
    
    # Uploads (Deep Read)
    # st.subheader("📄 Uploads")
    uploaded_file = st.file_uploader("Upload Document", type=["pdf", "docx", "xlsx", "csv", "txt", "md"])    
    if uploaded_file and st.button("Process Upload"):
        with st.spinner("Ingesting..."):
            index, count = process_uploaded_file(uploaded_file)
            st.session_state.vector_store = index
            st.success(f"Indexed {count} chunks.")

    st.divider()

    # Local Disk (The Scout)
    # st.subheader("📂 Local Disk Access")
    folder_path = st.text_input("Add Folder Path (e.g., C:/Projects)")
    if st.button("Grant Permission"):
        success, msg = st.session_state.disk_scout.add_path(folder_path)
        if success:
            st.success(msg)
        else:
            st.error(msg)
            
    # Show Active Permissions
    if st.session_state.disk_scout.allowed_paths:
        st.caption("✅ Active Folders:")
        for p in st.session_state.disk_scout.allowed_paths:
            st.code(str(p))

# main UI
st.subheader("Let's Reason! - Ask DocuSenseAI")
st.caption("Your AI-powered assistant who respects your privacy..")
st.divider()


# asks to select mode, from uploaded documents or local disk scout
search_mode = st.radio("Search Mode:", ["Uploaded Documents", "Local Disk Scout"], horizontal=True)
st.divider()

# query = st.text_input("What are you looking for?")
query = st.text_input("What are you looking for?")


if query and st.button("Ask AI"):
    
    # UPLOADED DOCUMENTS
    if search_mode == "Uploaded Documents":
        if st.session_state.vector_store:
            with st.spinner("Thinking..."):
                answer, sources = query_local_model(query, st.session_state.vector_store)
            st.markdown("### 🤖 Answer:")
            st.write(answer)
            st.divider()
            with st.expander("View Source Chunks"):
                for doc in sources:
                    st.info(doc.page_content)
        else:
            st.error("Please upload a document first.")

    # LOCAL DISK SCOUT
    elif search_mode == "Local Disk Scout":
        # We ask the LLM: "What file is the user looking for?"
        # If the user typed "Write the content as it is?", this might fail if they didn't specify a file.
        # But if they typed "Write the content of the budget file", it will extract "budget".
        
        # need to handle the case where the user is referring to previous results, somewhat like a chain of thought 
        # For now, just assuming every query is a fresh search
        
        from processor import extract_search_keyword
        
        with st.spinner("Deciding what to search for..."):
            keyword = extract_search_keyword(query)
            st.caption(f"🔍 Searching for files matching: **'{keyword}'**")

        # SCOUT
        with st.spinner(f"Scanning disk..."):
            matches = st.session_state.disk_scout.scout_files(keyword)
        
        if not matches:
            st.warning(f"No files found containing '{keyword}'. Try specifying the filename explicitly (e.g., 'Check the notes file').")
        else:
            st.success(f"Found {len(matches)} files.")
            
            # read and reason..
            file_contents = []
            for m in matches:
                content = st.session_state.disk_scout.read_file_lazy(m)
                # limited content size avoids crashing 
                file_contents.append(f"FILENAME: {m.name}\nCONTENT: {content[:4000]}...")
            
            with st.spinner("Reading & Generating Answer..."):
                import ollama
                context_text = "\n\n".join(file_contents)
                
                system_prompt = f"""
                You are DocuSenseAI. 
                The user has asked a question about these specific local files.
                
                USER INSTRUCTION: {query}
                
                FILES FOUND:
                {context_text}
                
                Instructions:
                - If the user asks to "write the content", output the file content verbatim.
                - If the user asks for a summary, summarize.
                - Explicitly mention which file you are reading.
                """
                
                response = ollama.chat(model='phi3', messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': "Execute the instruction based on the files above."},
                ])
                
                st.markdown("### 🤖 Local Insight:")
                st.write(response['message']['content'])
                
                st.divider()
                st.write("📂 **Files Accessed:**")
                for m in matches:
                    st.code(str(m))