import os
import streamlit as st
from rag_graph import run_agentic_rag

st.set_page_config(page_title="Agentic RAG: PDF + Web", page_icon="📄", layout="centered")
st.title("📄 Agentic RAG (PDF first, then Wikipedia/Web)")
st.write("Upload a PDF and ask a question. The system will use the PDF first; if insufficient, it searches Wikipedia or the web.")

with st.sidebar:
    st.header("Settings")
    openai_key = st.text_input("OPENAI_API_KEY", type="password", help="Required for LLM responses.")
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    # Optional: enable Tavily or SerpAPI for broader web search
    # tavily = st.text_input("TAVILY_API_KEY", type="password")
    # if tavily: os.environ["TAVILY_API_KEY"] = tavily
    # serp = st.text_input("SERPAPI_API_KEY", type="password")
    # if serp: os.environ["SERPAPI_API_KEY"] = serp

uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
query = st.text_input("Your question")
run = st.button("Get answer")

if run:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Please provide OPENAI_API_KEY in the sidebar.")
    elif not query:
        st.error("Please enter a question.")
    else:
        pdf_path = None
        if uploaded_pdf:
            tmp_dir = os.path.join(os.getcwd(), ".tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_pdf_path = os.path.join(tmp_dir, f"uploaded_{uploaded_pdf.name}")
            with open(tmp_pdf_path, "wb") as f:
                f.write(uploaded_pdf.getbuffer())
            pdf_path = tmp_pdf_path

        with st.spinner("Thinking..."):
            answer, prov = run_agentic_rag(query=query, pdf_path=pdf_path)

        st.subheader("Answer")
        st.write(answer)

        st.divider()
        st.subheader("Provenance")
        st.write(prov)
        if prov.get("source") == "pdf":
            st.info("Answered from the PDF context.")
        elif prov.get("source") == "agent":
            st.info("Answered using agentic tools (PDF retriever + Wikipedia/Web).")
