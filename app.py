import os
import streamlit as st
import tempfile
import google.generativeai as genai

from safeguarding_logic_native import load_and_split_pdf, create_vector_store

# --- Streamlit Layout ---
st.set_page_config(page_title="Safeguarding Gemini Assistant", layout="wide")
st.title("🏫 Safeguarding Assistant with Gemini (v1 API)")
st.caption("Gemini-powered AI Assistant trained on safeguarding policies")

# --- Check API Key ---
if 'GOOGLE_API_KEY' not in st.secrets:
    st.error("🚨 Google API Key not found in Streamlit secrets.")
    st.stop()

# --- Configure Gemini ---
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-pro")

# --- Session State ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Welcome! Please upload a safeguarding policy PDF to get started."
    }]

# --- Sidebar Upload ---
with st.sidebar:
    st.header("📁 Upload Policy PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file and st.button("Process Policy Document"):
        with st.spinner("Reading and embedding document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                pdf_path = tmp.name

            try:
                docs = load_and_split_pdf(pdf_path)
                if docs:
                    st.session_state.vector_store = create_vector_store(docs)
                    st.success("✅ Policy processed successfully. You can now ask questions.")
                    st.session_state.messages = [{
                        "role": "assistant",
                        "content": "The safeguarding policy has been loaded. How can I assist you?"
                    }]
                else:
                    st.error("⚠️ Could not extract content from PDF.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                os.remove(pdf_path)

# --- Chat Interface ---
st.header("💬 Ask About Safeguarding")

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask your question here..."):
    if not st.session_state.vector_store:
        st.error("⚠️ Please upload and process a safeguarding policy first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Use Gemini for answer generation with document context
        with st.spinner("Gemini is thinking..."):
            try:
                retriever = st.session_state.vector_store.as_retriever()
                docs = retriever.get_relevant_documents(prompt)
                context = "\n".join(doc.page_content for doc in docs[:3])
                final_prompt = f"Refer to the following safeguarding policy content:

{context}

Question: {prompt}"
                response = model.generate_content(final_prompt)
                answer = response.text
            except Exception as e:
                answer = "⚠️ Sorry, an error occurred."
                st.error(str(e))

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
