"""
EduLaptop Advisor — RAG Chatbot
CCAI 435 | University of Jeddah
Generation: Claude API (llama-3.3-70b-versatile)
"""

import streamlit as st
import pandas as pd
import time, re, os, sys


sys.path.insert(0, os.path.dirname(__file__))
from rag_engine import LaptopRAG
from ragas_evaluator import run_full_evaluation, evaluate_sample

st.set_page_config(page_title="EduLaptop Advisor", page_icon="🎓",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
html,body,[class*="css"]{font-family:'Segoe UI',sans-serif;}
.header{background:linear-gradient(135deg,#1a237e,#1565c0);padding:1.8rem 2rem;
  border-radius:16px;color:white;margin-bottom:1.5rem;box-shadow:0 4px 20px rgba(26,35,126,.3);}
.header h1{font-size:2rem;margin:0;font-weight:700;}
.header p{margin:.3rem 0 0;opacity:.88;}
.user-msg{background:#1565c0;color:white;padding:.85rem 1.2rem;
  border-radius:18px 18px 4px 18px;margin:.5rem 0 .5rem auto;max-width:72%;width:fit-content;}
.bot-msg{background:#f0f4ff;color:#1a1a2e;padding:.85rem 1.2rem;
  border-radius:18px 18px 18px 4px;margin:.5rem auto .5rem 0;
  max-width:92%;border-left:4px solid #1565c0;}
.chat-area{background:white;border-radius:16px;padding:1.2rem;
  min-height:430px;max-height:530px;overflow-y:auto;border:1px solid #e3e8f0;}
.card{background:#f8f9ff;border:1px solid #dde3f5;border-radius:12px;padding:1rem;text-align:center;}
.card .num{font-size:1.8rem;font-weight:700;color:#1565c0;}
.card .lbl{font-size:.82rem;color:#666;margin-top:2px;}
.src{background:#fff;border:1px solid #dde3f5;border-radius:10px;
  padding:.9rem;margin-bottom:.6rem;font-size:.84rem;}
.badge{background:#1565c0;color:white;border-radius:20px;padding:2px 10px;font-size:.78rem;}
.chat-box > div:first-child {
    max-height: 500px !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
}
.api-on{background:#e8f5e9;border:1px solid #a5d6a7;border-radius:10px;
  padding:.6rem 1rem;font-size:.88rem;color:#2e7d32;margin-bottom:.5rem;}
.api-off{background:#fff8e1;border:1px solid #ffe082;border-radius:10px;
  padding:.6rem 1rem;font-size:.88rem;color:#f57f17;margin-bottom:.5rem;}
</style>""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for k,v in [("messages",[]),("query_count",0),("last_result",None),
            ("eval_cache",None),("rag",None)]:
    if k not in st.session_state: st.session_state[k] = v

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 EduLaptop Advisor")
    st.markdown("**CCAI 435 — Deep Learning**")
    st.divider()

    page = st.radio("", ["💬 Chatbot","📊 Dashboard","🏗️ Architecture","📈 RAGAS Evaluation","👥 Team"])
    st.divider()

    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Top-K Results", 3, 6, 5)
    st.divider()

    st.markdown("### 💡 Sample Queries")
    for i, s in enumerate(["Gaming laptop with dedicated GPU",
              "Laptop with 16GB RAM for programming",
              "Best HP laptop under SAR 3000",
              "Lenovo laptop with Core i7 processor",
              "Laptop with Nvidia GPU for engineering",
              "Dell laptop with SSD 512GB"]):
        if st.button(s, use_container_width=True, key=f"sq_{i}"):
            st.session_state._quick = s
    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages=[]; st.session_state.query_count=0
        st.session_state.last_result=None; st.rerun()

# ── Load RAG ───────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@st.cache_resource(show_spinner="🔄 Loading 991 laptops into knowledge base…")
def load_rag_base():
    return LaptopRAG(api_key=GROQ_API_KEY)

if st.session_state.rag is None:
    st.session_state.rag = load_rag_base()

rag   = st.session_state.rag
stats = rag.get_stats()
df    = rag.df

# ══════════════════════════════════════════════════════════════════════════════
# CHATBOT
# ══════════════════════════════════════════════════════════════════════════════
if page == "💬 Chatbot":
    st.markdown("<div class='header'><h1>🎓 EduLaptop Advisor</h1>"
                "<p>Your Smart Laptop Advisor — Find the Perfect Laptop for Your Studies · Prices in SAR 🇸🇦</p></div>",
                unsafe_allow_html=True)

    chat_col, src_col = st.columns([3,1])

    with chat_col:
        st.markdown("### 💬 Chat")
        st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
        with st.container(border=True, height=500):
            if not st.session_state.messages:
                st.markdown(
                    "<div style='text-align:center;padding:3rem;color:#999;'>"
                    "<div style='font-size:3rem'>🎓</div>"
                    "<div style='font-size:1.1rem;font-weight:600;margin-top:.5rem;color:#555'>"
                    "Welcome to EduLaptop Advisor!</div>"
                    "<div style='font-size:.9rem;margin-top:.4rem'>"
                    "Ask about laptops for education · Prices in SAR 🇸🇦<br>"
                    "Try: <em>\"Best laptop for CS students\"</em></div></div>",
                    unsafe_allow_html=True)
            else:
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        st.markdown(
                            f"<div style='background:#1565c0;color:white;padding:.75rem 1rem;"
                            f"border-radius:18px 18px 4px 18px;margin:.5rem 0 .5rem auto;"
                            f"max-width:75%;width:fit-content;font-size:.95rem;'>👤 {msg['content']}</div>",
                            unsafe_allow_html=True)
                    else:
                        with st.container(border=True):
                            st.markdown(msg["content"])

        st.markdown("</div>", unsafe_allow_html=True)
        query_input = st.chat_input("Ask about laptops for education… (SAR 🇸🇦)")

    with src_col:
        st.markdown("### 📚 Top-5 Sources")
        if st.session_state.last_result:
            for r in st.session_state.last_result["retrieved"]:
                d   = r["data"]
                pct = min(int(r["score"]*300), 99)
                st.markdown(
                    f"<div class='src'><b>{str(d['brand']).title()}</b><br>"
                    f"<span style='font-size:.8rem;color:#444'>{str(d['Model'])[:38]}…</span><br>"
                    f"<span style='color:#1565c0;font-weight:600'>{d['Price_SAR']}</span>"
                    f" · ⭐{d['Rating']}<br>"
                    f"<span class='badge'>Match {pct}%</span></div>",
                    unsafe_allow_html=True)
        else:
            st.markdown("<div style='color:#aaa;font-size:.9rem;padding:1rem;text-align:center'>"
                        "Top-5 sources appear after your query.</div>", unsafe_allow_html=True)

    # Live RAGAS scores
    if st.session_state.last_result:
        r   = st.session_state.last_result
        ctx = [x["document"] for x in r["retrieved"]]
        sc  = evaluate_sample(r["query"], r["answer"], ctx)
        with st.expander("📊 RAGAS Scores for this query"):
            m1,m2,m3,m4 = st.columns(4)
            for col,lbl,val in [(m1,"Context Precision",sc["context_precision"]),
                                (m2,"Answer Relevancy", sc["answer_relevancy"]),
                                (m3,"Faithfulness",     sc["faithfulness"]),
                                (m4,"RAGAS Score",      sc["ragas_score"])]:
                color = "#2e7d32" if val>=0.7 else ("#f57f17" if val>=0.4 else "#c62828")
                col.markdown(f"<div class='card'><div class='num' style='color:{color}'>{val}</div>"
                             f"<div class='lbl'>{lbl}</div></div>", unsafe_allow_html=True)

    if hasattr(st.session_state,"_quick") and st.session_state._quick:
        query_input = st.session_state._quick; del st.session_state._quick
    if query_input and query_input.strip():
        q = query_input.strip()
        st.session_state.messages.append({"role":"user","content":q})
        label = "🤖 Asking Groq (Llama 3.3)…"
        with st.spinner(label):
            result = rag.answer(q, top_k=top_k)
        st.session_state.messages.append({"role":"assistant","content":result["answer"]})
        st.session_state.last_result  = result
        st.session_state.query_count += 1
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    st.markdown("<div class='header'><h1>📊 Knowledge Base Dashboard</h1>"
                "<p>991 Real Laptops · Kaggle Dataset · Prices in SAR 🇸🇦</p></div>",
                unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Laptops by Brand")
        st.bar_chart(df["brand"].str.title().value_counts())
    with c2:
        st.subheader("Laptops by Category")
        st.bar_chart(df["Category"].value_counts())
    c3,c4 = st.columns(2)
    with c3:
        st.subheader("GPU Type Distribution")
        st.bar_chart(df["gpu_type"].value_counts())
    with c4:
        st.subheader("OS Distribution")
        st.bar_chart(df["OS"].value_counts())
    st.subheader("Average Price by Category (SAR)")
    st.bar_chart(df.groupby("Category")["Price_SAR_num"].mean().sort_values())
    st.subheader("RAM Distribution")
    st.bar_chart(df["ram_memory"].value_counts().sort_index())
    st.subheader("📋 Full Dataset (991 laptops)")
    disp = df[["brand","Model","Category","Price_SAR","Rating","CPU_desc",
               "ram_memory","Storage","Display","GPU_desc","OS"]].copy()
    disp.columns = ["Brand","Model","Category","Price (SAR)","Rating",
                    "CPU","RAM (GB)","Storage","Display","GPU","OS"]
    st.dataframe(disp, use_container_width=True, height=400)

# ══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏗️ Architecture":
    st.markdown("<div class='header'><h1>🏗️ RAG System Architecture</h1>"
                "<p>Retrieval-Augmented Generation + Groq AI (Llama 3.3-70b)</p></div>", unsafe_allow_html=True)

    # Visual Pipeline
    st.components.v1.html("""
<svg width="100%" viewBox="0 0 680 1050" xmlns="http://www.w3.org/2000/svg" style="font-family:Segoe UI,sans-serif;">
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M2 1L8 5L2 9" fill="none" stroke="#555" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
  </marker>
</defs>
<text x="340" y="30" text-anchor="middle" font-size="17" font-weight="600" fill="#ffffff">Hybrid RAG Pipeline with Smart Filtering</text>
<line x1="60" y1="42" x2="620" y2="42" stroke="#ffffff" stroke-width="0.8" opacity="0.3"/>
<rect x="44" y="56" width="592" height="300" rx="14" fill="#EBF3FC" stroke="#185FA5" stroke-width="0.8"/>
<rect x="44" y="56" width="592" height="34" rx="14" fill="#185FA5" opacity="0.18"/>
<text x="340" y="78" text-anchor="middle" font-size="13" font-weight="700" fill="#0C447C">OFFLINE — Indexing Phase</text>
<rect x="66" y="102" width="36" height="36" rx="8" fill="#185FA5" opacity="0.12"/>
<text x="84" y="125" text-anchor="middle" font-size="20">🗄️</text>
<text x="116" y="116" font-size="13" font-weight="600" fill="#1a1a2e">1. Load Kaggle Dataset</text>
<text x="116" y="131" font-size="11" fill="#666">991 laptops · 26 brands · 22 features per laptop</text>
<line x1="60" y1="148" x2="620" y2="148" stroke="#aaa" stroke-width="0.5" stroke-dasharray="3 3"/>
<rect x="66" y="160" width="36" height="36" rx="8" fill="#185FA5" opacity="0.12"/>
<text x="84" y="183" text-anchor="middle" font-size="20">🧹</text>
<text x="116" y="174" font-size="13" font-weight="600" fill="#1a1a2e">2. Data Preprocessing &amp; SAR Conversion</text>
<text x="116" y="189" font-size="11" fill="#666">Cleaning · INR → SAR (×0.045) · derive Category, Best_For, CPU_desc, GPU_desc</text>
<line x1="60" y1="208" x2="620" y2="208" stroke="#aaa" stroke-width="0.5" stroke-dasharray="3 3"/>
<rect x="66" y="220" width="36" height="36" rx="8" fill="#185FA5" opacity="0.12"/>
<text x="84" y="243" text-anchor="middle" font-size="20">📄</text>
<text x="116" y="234" font-size="13" font-weight="600" fill="#1a1a2e">3. Build Documents</text>
<text x="116" y="249" font-size="11" fill="#666">Document Builder — each laptop row → rich natural language document</text>
<line x1="60" y1="268" x2="620" y2="268" stroke="#aaa" stroke-width="0.5" stroke-dasharray="3 3"/>
<rect x="66" y="280" width="36" height="36" rx="8" fill="#185FA5" opacity="0.12"/>
<text x="84" y="303" text-anchor="middle" font-size="20">📊</text>
<text x="116" y="294" font-size="13" font-weight="600" fill="#1a1a2e">4. Vectorize and Save</text>
<text x="116" y="309" font-size="11" fill="#666">TF-IDF Vectorizer (ngram 1-2, 8000 features, sublinear TF) → tfidf.pkl</text>
<line x1="340" y1="358" x2="340" y2="378" stroke="#555" stroke-width="1.5" marker-end="url(#arrow)"/>
<rect x="44" y="382" width="592" height="656" rx="14" fill="#EDFAF4" stroke="#0F6E56" stroke-width="0.8"/>
<rect x="44" y="382" width="592" height="34" rx="14" fill="#0F6E56" opacity="0.18"/>
<text x="340" y="404" text-anchor="middle" font-size="13" font-weight="700" fill="#085041">ONLINE — Query Phase</text>
<rect x="66" y="418" width="36" height="36" rx="8" fill="#0F6E56" opacity="0.12"/>
<text x="84" y="441" text-anchor="middle" font-size="20">💬</text>
<text x="116" y="432" font-size="13" font-weight="600" fill="#1a1a2e">5. User Query</text>
<text x="116" y="447" font-size="11" fill="#666">Natural language input via Streamlit chatbot</text>
<line x1="60" y1="466" x2="620" y2="466" stroke="#aaa" stroke-width="0.5" stroke-dasharray="3 3"/>
<rect x="66" y="478" width="36" height="36" rx="8" fill="#0F6E56" opacity="0.12"/>
<text x="84" y="501" text-anchor="middle" font-size="20">🔤</text>
<text x="116" y="492" font-size="13" font-weight="600" fill="#1a1a2e">6. Query Expansion <tspan font-size="11" font-weight="400" fill="#666">(domain synonyms)</tspan></text>
<rect x="116" y="502" width="460" height="20" rx="4" fill="#0F6E56" opacity="0.08"/>
<text x="124" y="516" font-size="11" fill="#0F6E56">"CS" → "programming, coding, software development, computer science"</text>
<line x1="60" y1="536" x2="620" y2="536" stroke="#aaa" stroke-width="0.5" stroke-dasharray="3 3"/>
<rect x="66" y="548" width="36" height="36" rx="8" fill="#BA7517" opacity="0.12"/>
<text x="84" y="571" text-anchor="middle" font-size="20">🤖</text>
<text x="116" y="562" font-size="13" font-weight="600" fill="#1a1a2e">7. Intent &amp; Smart Filtering <tspan font-size="11" font-weight="400" fill="#666">(Groq API)</tspan></text>
<rect x="116" y="572" width="460" height="20" rx="4" fill="#BA7517" opacity="0.10"/>
<text x="124" y="586" font-size="11" fill="#BA7517">{max_price_sar,  min_ram_gb,  gpu_type,  processor_tier,  os...}</text>
<line x1="60" y1="606" x2="620" y2="606" stroke="#aaa" stroke-width="0.5" stroke-dasharray="3 3"/>
<rect x="66" y="618" width="36" height="36" rx="8" fill="#0F6E56" opacity="0.12"/>
<text x="84" y="641" text-anchor="middle" font-size="20">🔍</text>
<text x="116" y="632" font-size="13" font-weight="600" fill="#1a1a2e">8. Filtering &amp; Similarity</text>
<text x="116" y="647" font-size="11" fill="#666">Apply JSON filters → TF-IDF Transform + Cosine Similarity on filtered set</text>
<line x1="60" y1="666" x2="620" y2="666" stroke="#aaa" stroke-width="0.5" stroke-dasharray="3 3"/>
<rect x="66" y="678" width="36" height="36" rx="8" fill="#0F6E56" opacity="0.12"/>
<text x="84" y="701" text-anchor="middle" font-size="20">📋</text>
<text x="116" y="692" font-size="13" font-weight="600" fill="#1a1a2e">9. Document Retrieval</text>
<text x="116" y="707" font-size="11" fill="#666">Fetch Top-5 most relevant laptop documents ranked by similarity score</text>
<line x1="60" y1="726" x2="620" y2="726" stroke="#aaa" stroke-width="0.5" stroke-dasharray="3 3"/>
<rect x="66" y="738" width="36" height="36" rx="8" fill="#534AB7" opacity="0.12"/>
<text x="84" y="761" text-anchor="middle" font-size="20">⚡</text>
<text x="116" y="752" font-size="13" font-weight="600" fill="#1a1a2e">10. Answer Generation <tspan font-size="11" font-weight="400" fill="#666">(Groq API)</tspan></text>
<text x="116" y="767" font-size="11" fill="#666">llama-3.3-70b-versatile · structured table + short recommendation · SAR prices</text>
<line x1="60" y1="786" x2="620" y2="786" stroke="#aaa" stroke-width="0.5" stroke-dasharray="3 3"/>
<rect x="66" y="798" width="36" height="36" rx="8" fill="#993C1D" opacity="0.12"/>
<text x="84" y="821" text-anchor="middle" font-size="20">📈</text>
<text x="116" y="812" font-size="13" font-weight="600" fill="#1a1a2e">11. RAGAS Evaluation <tspan font-size="11" font-weight="400" fill="#666">(live per query)</tspan></text>
<rect x="116" y="822" width="130" height="20" rx="4" fill="#993C1D" opacity="0.08"/>
<text x="124" y="836" font-size="11" fill="#993C1D">• Context Precision</text>
<rect x="258" y="822" width="128" height="20" rx="4" fill="#993C1D" opacity="0.08"/>
<text x="266" y="836" font-size="11" fill="#993C1D">• Answer Relevancy</text>
<rect x="398" y="822" width="100" height="20" rx="4" fill="#993C1D" opacity="0.08"/>
<text x="406" y="836" font-size="11" fill="#993C1D">• Faithfulness</text>
<line x1="60" y1="854" x2="620" y2="854" stroke="#aaa" stroke-width="0.5" stroke-dasharray="3 3"/>
<rect x="66" y="866" width="36" height="36" rx="8" fill="#0F6E56" opacity="0.12"/>
<text x="84" y="889" text-anchor="middle" font-size="20">🖥️</text>
<text x="116" y="880" font-size="13" font-weight="600" fill="#1a1a2e">12. Streamlit UI</text>
<text x="116" y="895" font-size="11" fill="#666">5-page web app · chatbot · dashboard · architecture · RAGAS · team</text>
</svg>
""", height=1500, scrolling=False)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#f8f9ff;border:1px solid #dde3f5;border-radius:14px;padding:1.5rem;margin-bottom:1.5rem;'>
    <div style='font-size:1.1rem;font-weight:700;color:#1a237e;margin-bottom:1rem;'>⚙️ Component Table</div>
    <table style='width:100%;border-collapse:collapse;font-size:.9rem;'>
      <thead>
        <tr style='background:#1565c0;color:white;'>
          <th style='padding:10px 14px;text-align:left;border-radius:8px 0 0 0;'>Component</th>
          <th style='padding:10px 14px;text-align:left;'>Technology</th>
          <th style='padding:10px 14px;text-align:left;border-radius:0 8px 0 0;'>Detail</th>
        </tr>
      </thead>
      <tbody>
        <tr style='background:#fff;'><td style='padding:9px 14px;border-bottom:1px solid #eee;color:#111;'>Dataset</td><td style='padding:9px 14px;border-bottom:1px solid #eee;color:#111;'>Kaggle Laptop Dataset</td><td style='padding:9px 14px;border-bottom:1px solid #eee;color:#111;'>991 laptops, 26 brands, 22 features</td></tr>
        <tr style='background:#f8f9ff;'><td style='padding:9px 14px;border-bottom:1px solid #eee;color:#111;'>Currency</td><td style='padding:9px 14px;border-bottom:1px solid #eee;color:#111;'>INR → SAR</td><td style='padding:9px 14px;border-bottom:1px solid #eee;color:#111;'>1 INR = 0.045 SAR</td></tr>
        <tr style='background:#fff;'><td style='padding:9px 14px;border-bottom:1px solid #eee;color:#111;'>Document Builder</td><td style='padding:9px 14px;border-bottom:1px solid #eee;color:#111;'>Python / Pandas</td><td style='padding:9px 14px;border-bottom:1px solid #eee;color:#111;'>Each laptop row → natural language document</td></tr>
        <tr style='background:#f8f9ff;'><td style='padding:9px 14px;border-bottom:1px solid #eee;color:#111;'>Vectorization</td><td style='padding:9px 14px;border-bottom:1px solid #eee;color:#111;'>TF-IDF (scikit-learn)</td><td style='padding:9px 14px;border-bottom:1px solid #eee;color:#111;'>ngram(1,2), 8000 features, sublinear TF</td></tr>
        <tr style='background:#fff;'><td style='padding:9px 14px;border-bottom:1px solid #eee;color:#111;'>Query Expansion</td><td style='padding:9px 14px;border-bottom:1px solid #eee;color:#111;'>Keyword Dictionary</td><td style='padding:9px 14px;border-bottom:1px solid #eee;color:#111;'>Appends domain synonyms to improve recall</td></tr>
        <tr style='background:#f8f9ff;'><td style='padding:9px 14px;border-bottom:1px solid #eee;font-weight:700;color:#1565c0;'>Smart Filtering</td><td style='padding:9px 14px;border-bottom:1px solid #eee;font-weight:700;color:#1565c0;'>Groq API + JSON</td><td style='padding:9px 14px;border-bottom:1px solid #eee;font-weight:700;color:#1565c0;'>Extracts constraints from query, filters dataset</td></tr>
        <tr style='background:#fff;'><td style='padding:9px 14px;border-bottom:1px solid #eee;color:#111;'>Retrieval</td><td style='padding:9px 14px;border-bottom:1px solid #eee;color:#111;'>Cosine Similarity</td><td style='padding:9px 14px;border-bottom:1px solid #eee;color:#111;'>Top-5 from filtered candidates</td></tr>
        <tr style='background:#f8f9ff;'><td style='padding:9px 14px;border-bottom:1px solid #eee;font-weight:700;color:#1565c0;'>Generation</td><td style='padding:9px 14px;border-bottom:1px solid #eee;font-weight:700;color:#1565c0;'>Groq API (Llama 3.3)</td><td style='padding:9px 14px;border-bottom:1px solid #eee;font-weight:700;color:#1565c0;'>llama-3.3-70b-versatile</td></tr>
        <tr style='background:#fff;'><td style='padding:9px 14px;border-bottom:1px solid #eee;color:#111;'>Fallback</td><td style='padding:9px 14px;border-bottom:1px solid #eee;color:#111;'>Rule-based templates</td><td style='padding:9px 14px;border-bottom:1px solid #eee;color:#111;'>If API unavailable</td></tr>
        <tr style='background:#f8f9ff;'><td style='padding:9px 14px;border-bottom:1px solid #eee;color:#111;'>Evaluation</td><td style='padding:9px 14px;border-bottom:1px solid #eee;color:#111;'>RAGAS (non-LLM)</td><td style='padding:9px 14px;border-bottom:1px solid #eee;color:#111;'>Context Precision, Answer Relevancy, Faithfulness</td></tr>
        <tr style='background:#fff;'><td style='padding:9px 14px;border-bottom:1px solid #eee;color:#111;'>Interface</td><td style='padding:9px 14px;border-bottom:1px solid #eee;color:#111;'>Streamlit</td><td style='padding:9px 14px;border-bottom:1px solid #eee;color:#111;'>5-page web app</td></tr>
        <tr style='background:#f8f9ff;'><td style='padding:9px 14px;color:#111;'>IDE</td><td style='padding:9px 14px;color:#111;'>VS Code</td><td style='padding:9px 14px;color:#111;'>launch.json configuration included</td></tr>
      </tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#f0f4ff;border:1px solid #c5cae9;border-radius:14px;padding:1.5rem;'>
    <div style='font-size:1.1rem;font-weight:700;color:#1a237e;margin-bottom:1rem;'>🤖 Why Groq for Generation?</div>
    <table style='width:100%;border-collapse:collapse;font-size:.9rem;'>
      <thead>
        <tr style='background:#1565c0;color:white;'>
          <th style='padding:10px 14px;text-align:left;border-radius:8px 0 0 0;'>Metric</th>
          <th style='padding:10px 14px;text-align:center;'>Rule-based</th>
          <th style='padding:10px 14px;text-align:center;border-radius:0 8px 0 0;'>Groq API</th>
        </tr>
      </thead>
      <tbody>
        <tr style='background:#fff;'><td style='padding:9px 14px;border-bottom:1px solid #eee;color:#111;'>Answer Relevancy</td><td style='padding:9px 14px;border-bottom:1px solid #eee;text-align:center;color:#f57f17;'>~0.58</td><td style='padding:9px 14px;border-bottom:1px solid #eee;text-align:center;color:#2e7d32;font-weight:700;'>~0.85+</td></tr>
        <tr style='background:#f8f9ff;'><td style='padding:9px 14px;border-bottom:1px solid #eee;color:#111;'>RAGAS Score</td><td style='padding:9px 14px;border-bottom:1px solid #eee;text-align:center;color:#f57f17;'>~0.75</td><td style='padding:9px 14px;border-bottom:1px solid #eee;text-align:center;color:#2e7d32;font-weight:700;'>~0.87+</td></tr>
        <tr style='background:#fff;'><td style='padding:9px 14px;color:#111;'>Response Style</td><td style='padding:9px 14px;text-align:center;color:#888;'>Fixed template</td><td style='padding:9px 14px;text-align:center;color:#2e7d32;font-weight:700;'>Contextual & intelligent</td></tr>
      </tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# RAGAS EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 RAGAS Evaluation":
    st.markdown("<div class='header'><h1>📈 RAGAS Evaluation</h1>"
                "<p>Context Precision · Answer Relevancy · Faithfulness</p></div>",
                unsafe_allow_html=True)

    mode = "🤖 Groq API (Llama 3.3-70b)"
    st.info(f"""**Generation mode:** {mode}

**RAGAS Metrics** (non-LLM) evaluated on 6 educational test queries:
- **Context Precision** — Are the most relevant laptops ranked at the top? *(Weighted Precision@K)*
- **Answer Relevancy** — How pertinent is the answer to the query? *(TF-IDF cosine + keyword overlap)*
- **Faithfulness** — Is the answer grounded in retrieved documents? *(Token overlap on factual claims)*

📚 Reference: [docs.ragas.io/en/latest/concepts/metrics/available_metrics](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/)""")

    if st.button("▶ Run RAGAS Evaluation", type="primary", use_container_width=True):
        with st.spinner("Evaluating 6 test queries…"):
            st.session_state.eval_cache = run_full_evaluation(rag)

    if st.session_state.eval_cache:
        ev  = st.session_state.eval_cache
        avg = ev["averages"]
        st.markdown("---")
        st.subheader("📊 Average Scores")
        c1,c2,c3,c4 = st.columns(4)
        for col,lbl,val in [(c1,"Context Precision",avg["avg_context_precision"]),
                            (c2,"Answer Relevancy", avg["avg_answer_relevancy"]),
                            (c3,"Faithfulness",     avg["avg_faithfulness"]),
                            (c4,"RAGAS Score",      avg["avg_ragas_score"])]:
            color = "#2e7d32" if val>=0.7 else ("#f57f17" if val>=0.4 else "#c62828")
            col.markdown(f"<div class='card'><div class='num' style='color:{color}'>{val}</div>"
                         f"<div class='lbl'>{lbl}</div></div>", unsafe_allow_html=True)

        st.subheader("🔍 Detailed Results")
        for s in ev["samples"]:
            sc   = s["scores"]
            icon = "🟢" if sc["ragas_score"]>=0.7 else ("🟡" if sc["ragas_score"]>=0.4 else "🔴")
            with st.expander(f"{icon} {s['query']}"):
                m1,m2,m3,m4 = st.columns(4)
                for col,lbl,val in [(m1,"Context Precision",sc["context_precision"]),
                                    (m2,"Answer Relevancy", sc["answer_relevancy"]),
                                    (m3,"Faithfulness",     sc["faithfulness"]),
                                    (m4,"RAGAS Score",      sc["ragas_score"])]:
                    color = "#2e7d32" if val>=0.7 else ("#f57f17" if val>=0.4 else "#c62828")
                    col.markdown(f"<div class='card'><div class='num' style='color:{color}'>{val}</div>"
                                 f"<div class='lbl'>{lbl}</div></div>", unsafe_allow_html=True)
                st.markdown("**Top-5 Retrieved Laptops:**")
                for i,name in enumerate(s["retrieved"],1):
                    st.markdown(f"{i}. {name}")
    else:
        st.markdown("<div style='text-align:center;padding:3rem;color:#aaa'>"
                    "Click <b>▶ Run RAGAS Evaluation</b> to start.</div>",
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TEAM
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Team":
    st.markdown("<div class='header'><h1>👥 Project Team</h1>"
                "<p>CCAI 435 — Deep Learning · University of Jeddah · College of Computer Science and Engineering · Department of Artificial Intelligence</p></div>",
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, name, sid in [
        (c1, "Renad Alharbi", "2210625"),
        (c2, "Shouq Ahyaf",  "2211559"),
        (c3, "Jood Khamjan", "2210846"),
    ]:
        col.markdown(f"""
        <div style='background:#f8f9ff;border:1px solid #dde3f5;border-radius:16px;
        padding:2rem 1rem;text-align:center;'>
            <div style='font-size:3.5rem;margin-bottom:.75rem;'>👩‍💻</div>
            <div style='font-size:1.15rem;font-weight:700;color:#1a237e;'>{name}</div>
            <div style='font-size:.88rem;color:#1565c0;margin-top:.4rem;'>Student ID: {sid}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Task Distribution
    st.markdown("""
    <div style='background:#f8f9ff;border:1px solid #dde3f5;border-radius:14px;padding:1.5rem;margin-bottom:1.5rem;'>
    <div style='font-size:1.1rem;font-weight:700;color:#1a237e;margin-bottom:1rem;'>📋 Task Distribution</div>
    <table style='width:100%;border-collapse:collapse;font-size:.95rem;'>
      <thead>
        <tr style='background:#1565c0;color:white;'>
          <th style='padding:12px 16px;text-align:left;border-radius:8px 0 0 0;width:160px;'>Member</th>
          <th style='padding:12px 16px;text-align:left;border-radius:0 8px 0 0;'>Responsibilities</th>
        </tr>
      </thead>
      <tbody>
        <tr style='background:#fff;'>
          <td style='padding:12px 16px;border-bottom:1px solid #eee;font-weight:700;color:#1a237e;font-size:1rem;'>Shouq</td>
          <td style='padding:12px 16px;border-bottom:1px solid #eee;font-size:.95rem;color:#333;'>Dataset preprocessing, feature engineering, TF-IDF vectorization, and retrieval pipeline implementation.</td>
        </tr>
        <tr style='background:#f8f9ff;'>
          <td style='padding:12px 16px;border-bottom:1px solid #eee;font-weight:700;color:#1a237e;font-size:1rem;'>Renad</td>
          <td style='padding:12px 16px;border-bottom:1px solid #eee;font-size:.95rem;color:#333;'>Groq API integration, smart filtering mechanism, Streamlit interface development, and system deployment.</td>
        </tr>
        <tr style='background:#fff;'>
          <td style='padding:12px 16px;font-weight:700;color:#1a237e;font-size:1rem;'>Jood</td>
          <td style='padding:12px 16px;font-size:.95rem;color:#333;'>RAGAS evaluation implementation, experimental analysis, report writing, and documentation.</td>
        </tr>
      </tbody>
    </table>
    <div style='font-size:.92rem;color:#555;margin-top:1rem;line-height:1.7;'>
    In addition, all team members participated in system testing, debugging, architecture design discussions, and final presentation preparation.
    </div>
    </div>
    """, unsafe_allow_html=True)