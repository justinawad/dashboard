
# Market_Visualizer — Decision Support Dashboard (Streamlit + ML + GenAI)

**Market_Visualizer** is a decision-support dashboard designed for **candidates and recruiters**.  
It makes a salary prediction model usable in real life, and adds **explainability, market context, and GenAI assistance** through a production-oriented interface.

## What the app delivers
### Candidate side
- **Salary prediction** from a job description (ML inference with a trained XGBoost model)
- **Market & geography context** (interactive visualizations: maps, distributions, comparisons)
- **Skill extraction & explanation** (GenAI extracts structured hard skills and explains drivers)
- **Career Coach (RAG)**: upload a CV and ask questions grounded on **verifiable sources** (CV chunks + job offer context), reducing hallucinations

### Recruiter side (mass screening at scale)
- Upload dozens of CVs and get an **instant ranking** by **semantic similarity**
- Fast and scalable: ranking uses **local embeddings** (no LLM cost/latency for the first pass)
- Optional deep analysis (GenAI) on top profiles only (“why this candidate?”)

## Technical architecture (high level)
- **UI + Backend**: Streamlit (Python)
- **Prediction engine**: XGBoost model loaded from `salary_model_xgboost.pkl`
- **Visualization**: Plotly (interactive charts & maps)
- **GenAI**: LLM (Gemini / Groq Llama 3.3 70B, depending on configuration) for skill extraction + chatbot
- **Local NLP**: `sentence-transformers` (all-MiniLM-L6-v2) for embeddings
- **Vector store**: FAISS (in-memory) for CV RAG and retrieval

## Performance & reliability
Streamlit reruns the script on each interaction, so caching is critical:
- `st.cache_resource`: loads heavy models (XGBoost + embedding model) once
- `st.cache_data`: caches LLM outputs to avoid repeated API calls
- `st.session_state`: keeps chat history and user context across reruns

## Key ideas behind the design
- **Hybrid AI**: classic ML for accurate numeric prediction + LLM for semantic reasoning
- **RAG grounded answers**: responses use retrieved CV chunks + job offer context to limit hallucinations
- **Scalable screening**: recruiter ranking is fast and cost-effective because it relies on local embeddings

## Demo-ready / “production-friendly”
The app is designed to be usable for demonstrations without infrastructure costs:
- Cloud LLM for reasoning + local embeddings for retrieval and ranking
- Robust parsing of structured outputs (JSON extraction via LangChain parsers when used)

