# bootstrap.py
import os

PROJECT = "agentic_rag_pdf"
FILES = {
    "README.md": """# Agentic RAG for PDF with LangGraph, LangChain, and Streamlit

This app:
- Uploads a PDF
- Answers queries grounded in the PDF first
- If insufficient, falls back to Wikipedia or optional web search
- Uses Streamlit for UI
- Uses LangChain for the agent and tools, orchestrated by LangGraph

## Quickstart
1. python bootstrap.py  (you already did this to create files)
2. cd agentic_rag_pdf
3. pip install -r requirements.txt
4. streamlit run app.py

Set your OPENAI_API_KEY in the sidebar before asking questions.

## Optional web search
Uncomment Tavily or SerpAPI in rag_graph.py and set TAVILY_API_KEY or SERPAPI_API_KEY in the sidebar.
""",
    "requirements.txt": """streamlit>=1.33.0
langchain>=0.2.14
langgraph>=0.2.30
langchain-community>=0.2.12
langchain-openai>=0.1.20
faiss-cpu>=1.8.0
pypdf>=4.0.2
numpy>=1.26.0
tiktoken>=0.7.0
wikipedia>=1.4.0
python-dotenv>=1.0.1
""",
    "app.py": """import os
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
""",
    "rag_graph.py": """import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.schema import Document
from langchain.tools import Tool
from langchain_community.utilities import WikipediaAPIWrapper
# Optional: Tavily or SerpAPI web search
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_community.tools.serpapi import SerpAPIWrapper

from langchain.agents import AgentType, initialize_agent
from langgraph.graph import StateGraph, END

# -------------------------
# LLM and embeddings
# -------------------------
def get_llm():
    # Requires OPENAI_API_KEY in environment
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

def get_embeddings():
    return OpenAIEmbeddings()

# -------------------------
# PDF ingestion and retriever
# -------------------------
@dataclass
class PDFIndex:
    vs: FAISS
    metadata: Dict[str, Any] = field(default_factory=dict)

def build_pdf_index(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> PDFIndex:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\\n\\n", "\\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    embeddings = get_embeddings()
    vs = FAISS.from_documents(chunks, embeddings)
    return PDFIndex(vs=vs, metadata={"pdf_path": pdf_path, "chunks": len(chunks)})

def retrieve_from_pdf(index: PDFIndex, query: str, k: int = 5) -> List[Document]:
    return index.vs.similarity_search(query, k=k)

def score_retrieval(retrieved: List[Document], query: str) -> float:
    if not retrieved:
        return 0.0
    lengths = [len(d.page_content) for d in retrieved]
    return min(1.0, (sum(lengths) / (len(lengths) * 1200.0)))  # normalize ~chunk_size

# -------------------------
# Tools: PDF retriever, Wikipedia, Web search
# -------------------------
def make_tools(pdf_index: Optional[PDFIndex]) -> List[Tool]:
    tools = []

    if pdf_index:
        def pdf_tool_fn(q: str) -> str:
            docs = retrieve_from_pdf(pdf_index, q, k=6)
            joined = "\\n\\n".join([f"Source[{i+1}]: {d.page_content}" for i, d in enumerate(docs)])
            return joined if joined else "No relevant passages found in the PDF."
        tools.append(Tool(
            name="pdf_retriever",
            description="Find passages directly from the uploaded PDF relevant to the query.",
            func=pdf_tool_fn,
        ))

    wiki = WikipediaAPIWrapper(language="en")
    tools.append(Tool(
        name="wikipedia",
        description="Search and read summaries from Wikipedia when PDF does not contain the answer.",
        func=lambda q: wiki.run(q),
    ))

    # Optional: broader web search
    # tavily_tool = TavilySearchResults()
    # tools.append(tavily_tool)
    # serpapi = SerpAPIWrapper()
    # tools.append(Tool(name="google_search", description="General web search via Google (SerpAPI).", func=serpapi.run))

    return tools

# -------------------------
# Agent: LangChain ReAct agent using tools
# -------------------------
def build_agent(tools: List[Tool]):
    llm = get_llm()
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=5,
        return_intermediate_steps=False,
    )
    return agent

# -------------------------
# LangGraph state and nodes
# -------------------------
@dataclass
class RAGState:
    query: str
    pdf_index: Optional[PDFIndex] = None
    retrieved: List[Document] = field(default_factory=list)
    retrieval_score: float = 0.0
    answer: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

def node_retrieve(state: RAGState) -> RAGState:
    if state.pdf_index:
        state.retrieved = retrieve_from_pdf(state.pdf_index, state.query, k=6)
        state.retrieval_score = score_retrieval(state.retrieved, state.query)
        state.provenance["retrieval_k"] = len(state.retrieved)
    else:
        state.retrieved = []
        state.revenance = 0.0
    return state

def node_answer_from_pdf(state: RAGState) -> RAGState:
    llm = get_llm()
    context = "\\n\\n".join([d.page_content for d in state.retrieved])
    prompt = (
        "You are a helpful assistant. Use ONLY the provided context to answer the question.\\n\\n"
        f"Context:\\n{context}\\n\\n"
        f"Question: {state.query}\\n\\n"
        "If the answer is not in the context, say 'Insufficient information in PDF context.'"
    )
    resp = llm.invoke(prompt)
    state.answer = resp.content
    state.provenance["source"] = "pdf"
    return state

def node_agentic_fallback(state: RAGState) -> RAGState:
    tools = make_tools(state.pdf_index)
    agent = build_agent(tools)
    resp = agent.run(state.query)
    state.answer = resp
    state.provenance["source"] = "agent"
    return state

def router_decision(state: RAGState) -> str:
    threshold = 0.25
    if state.retrieval_score >= threshold and len(state.retrieved) > 0:
        return "answer_from_pdf"
    return "agentic_fallback"

def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", node_retrieve)
    graph.add_node("answer_from_pdf", node_answer_from_pdf)
    graph.add_node("agentic_fallback", node_agentic_fallback)

    graph.set_entry_point("retrieve")

    graph.add_conditional_edges(
        "retrieve",
        router_decision,
        {
            "answer_from_pdf": "answer_from_pdf",
            "agentic_fallback": "agentic_fallback",
        },
    )

    graph.add_edge("answer_from_pdf", END)
    graph.add_edge("agentic_fallback", END)
    return graph.compile()

def run_agentic_rag(query: str, pdf_path: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    pdf_index = build_pdf_index(pdf_path) if pdf_path else None
    graph = build_graph()
    final = graph.invoke(RAGState(query=query, pdf_index=pdf_index))
    return final.answer or "No answer generated.", final.provenance
""",
    ".streamlit/config.toml": """[theme]
primaryColor="#4F46E5"
backgroundColor="#FFFFFF"
secondaryBackgroundColor="#F3F4F6"
textColor="#111827"
font="sans serif"
"""
}

def main():
    os.makedirs(PROJECT, exist_ok=True)
    for path, content in FILES.items():
        full = os.path.join(PROJECT, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
    print(f"Project '{PROJECT}' created.")
    print("Next steps:")
    print(f"  1. cd {PROJECT}")
    print("  2. pip install -r requirements.txt")
    print("  3. streamlit run app.py")

if __name__ == "__main__":
    main()
