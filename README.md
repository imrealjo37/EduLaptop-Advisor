# 🎓 EduLaptop Advisor — RAG Chatbot
**CCAI 435 Final Project | University of Jeddah**  
Domain: **Education** | System: **Retrieval-Augmented Generation (RAG)**

---

## 📌 Problem Statement
Students and educators struggle to find the right laptop that matches their academic needs, budget, and technical requirements. This RAG-based chatbot solves that by enabling natural language queries over a curated laptop knowledge base.

## 🏗️ System Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────┐
│              RAG PIPELINE                        │
│                                                  │
│  1. Query Embedding (all-MiniLM-L6-v2)          │
│       │                                          │
│       ▼                                          │
│  2. FAISS Vector Search (cosine similarity)      │
│       │                                          │
│       ▼                                          │
│  3. Top-K Document Retrieval                     │
│       │                                          │
│       ▼                                          │
│  4. Grounded Answer Generation                   │
└─────────────────────────────────────────────────┘
    │
    ▼
Structured Answer + Source Documents
```

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit app
```bash
streamlit run app.py
```

### 3. Open browser
Navigate to `http://localhost:8501`

---

## 📁 Project Structure
```
rag_project/
├── app.py              # Streamlit chatbot UI
├── rag_engine.py       # RAG pipeline (embed, index, retrieve, generate)
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── data/
    ├── laptops.csv     # Laptop knowledge base (32 laptops, 12 features)
    ├── faiss.index     # FAISS vector index (auto-generated)
    └── documents.pkl   # Processed documents (auto-generated)
```

## 📊 Dataset
- **Source:** Based on Kaggle Laptop Dataset (pradeepjangirml007/laptop-data-set)
- **Size:** 32 laptops from 11 brands
- **Features:** Brand, Model, Processor, RAM, Storage, Display, GPU, Battery, Weight, Price, Category, Best_For
- **Preprocessing:** Each row converted to a rich natural-language document for embedding

## 🧠 Models & Tools
| Component | Tool |
|-----------|------|
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Index | FAISS (IndexFlatIP with cosine similarity) |
| UI Framework | Streamlit |
| Data Processing | Pandas + NumPy |

## 📈 Evaluation
- **Retrieval Quality:** Cosine similarity scores shown per result
- **Coverage:** 32 laptops across 6 categories (Budget, Mid-Range, Premium, High-End, Gaming, Business)
- **Response Grounding:** All answers are grounded in retrieved documents

## 👥 Team
CCAI 435 — Deep Learning Course  
College of Computer Science and Engineering  
University of Jeddah, Kingdom of Saudi Arabia
