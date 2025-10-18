# Agentic RAG for PDF with LangGraph, LangChain, and Streamlit

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
