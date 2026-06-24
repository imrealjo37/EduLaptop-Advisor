"""
RAG Engine — EduLaptop Advisor
Dataset: Kaggle Laptop Dataset (991 laptops)
Generation: Groq API (llama-3.3-70b-versatile) — FREE
Prices: INR → SAR
"""

import pandas as pd
import pickle, os, re, json
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "data", "laptops_kaggle.csv")
IDX_PATH  = os.path.join(BASE, "data", "tfidf.pkl")

INR_TO_SAR = 0.045

def inr_to_sar(val) -> str:
    try:    return f"SAR {float(val)*INR_TO_SAR:,.0f}"
    except: return str(val)

def load_and_clean(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["Price_SAR"]     = df["Price"].apply(inr_to_sar)
    df["Price_SAR_num"] = df["Price"].apply(lambda x: round(float(x)*INR_TO_SAR, 0))
    df["Storage"] = df.apply(lambda r:
        f"{int(r['primary_storage_capacity'])}GB {r['primary_storage_type']}" +
        (f" + {int(r['secondary_storage_capacity'])}GB {r['secondary_storage_type']}"
         if str(r['secondary_storage_type']) not in ['No secondary storage','0','nan']
         and str(r['secondary_storage_capacity']) not in ['0','nan'] else ""), axis=1)
    df["GPU_desc"] = df.apply(lambda r:
        f"{str(r['gpu_brand']).title()} {'Dedicated' if r['gpu_type']=='dedicated' else 'Integrated'}", axis=1)
    df["Display"]  = df.apply(lambda r:
        f"{r['display_size']} inch {int(r['resolution_width'])}x{int(r['resolution_height'])}", axis=1)
    df["CPU_desc"] = df.apply(lambda r:
        f"{str(r['processor_brand']).title()} {str(r['processor_tier']).title()} ({int(r['num_cores'])} cores)", axis=1)
    df["Category"] = df.apply(_categorize, axis=1)
    df["Best_For"] = df.apply(_best_for, axis=1)
    return df.reset_index(drop=True)

def _categorize(row):
    price = float(row["Price"]) * INR_TO_SAR
    gpu   = str(row["gpu_type"]).lower()
    brand = str(row["brand"]).lower()
    if brand == "apple":                    return "Apple / MacBook"
    if gpu == "dedicated" and price > 3750: return "Gaming / High-Performance"
    if price < 1500:                        return "Budget"
    if price < 2800:                        return "Mid-Range"
    if price < 5000:                        return "Premium"
    return "High-End"

def _best_for(row):
    gpu   = str(row["gpu_type"]).lower()
    ram   = int(row["ram_memory"])
    tier  = str(row["processor_tier"]).lower()
    brand = str(row["brand"]).lower()
    price = float(row["Price"]) * INR_TO_SAR
    touch = str(row["is_touch_screen"]).lower() == "true"
    tags  = []
    if brand == "apple":                   tags.append("MacOS users and creative students")
    if gpu == "dedicated":                 tags.append("gaming and engineering students")
    if ram >= 16:                          tags.append("programming and multitasking")
    if price < 1500:                       tags.append("budget-conscious students")
    if touch:                              tags.append("students needing touch interface")
    if "i5" in tier or "ryzen 5" in tier: tags.append("everyday university use")
    if "i7" in tier or "ryzen 7" in tier: tags.append("professionals and grad students")
    if "i3" in tier or "ryzen 3" in tier: tags.append("light tasks and basic coursework")
    if not tags:                           tags.append("general education use")
    return ", ".join(tags[:2])

def row_to_document(row):
    brand       = str(row["brand"]).title()
    model_clean = re.sub(r'\b(CS|IT|AI|UN|IN)\b', '', str(row["Model"]), flags=re.IGNORECASE).strip()
    doc = (
        f"{brand} laptop. model {model_clean}. "
        f"category {row['Category']}. best for {row['Best_For']}. "
        f"processor {row['CPU_desc']}. ram {int(row['ram_memory'])} gb. "
        f"storage {row['Storage']}. display {row['Display']}. gpu {row['GPU_desc']}. "
        f"os {row['OS']}. "
        f"touchscreen {'yes' if str(row['is_touch_screen']).lower()=='true' else 'no'}. "
        f"warranty {row['year_of_warranty']} year. price {row['Price_SAR']}. "
        f"rating {row['Rating']} out of 100. "
        f"{brand} {row['Category']} {row['Best_For']} {row['CPU_desc']}"
    )
    return doc.lower()

_EXPANSIONS = {
    "cs":           "computer science programming coding software development",
    "computer sci": "programming coding software development",
    "programming":  "coding software development ram ssd fast processor dedicated",
    "coding":       "programming software development ram ssd",
    "engineering":  "engineering programming dedicated gpu ram cores",
    "ml":           "machine learning artificial intelligence deep learning gpu",
    "gaming":       "gaming dedicated gpu nvidia amd high performance cores",
    "medical":      "medical lightweight portable long battery display",
    "budget":       "budget affordable cheap low price entry level",
    "affordable":   "budget cheap low price entry level",
    "cheap":        "budget affordable low price",
    "portable":     "lightweight thin portable battery long",
    "lightweight":  "portable thin battery long",
    "university":   "student study everyday university college",
    "student":      "university study everyday college",
    "macbook":      "apple mac macbook macos",
    "apple":        "apple macbook macos",
    "data science": "machine learning python programming gpu ram",
    "design":       "creative display resolution color gpu dedicated",
}

def expand_query(query: str) -> str:
    q, extra = query.lower(), []
    for kw, expansion in _EXPANSIONS.items():
        if kw in q:
            extra.append(expansion)
    return (q + " " + " ".join(extra)).strip() if extra else q

def build_index(force=False):
    if not force and os.path.exists(IDX_PATH):
        with open(IDX_PATH,"rb") as f: saved = pickle.load(f)
        df = load_and_clean(DATA_PATH)
        return saved["vectorizer"], saved["matrix"], saved["documents"], df
    df   = load_and_clean(DATA_PATH)
    docs = [row_to_document(row) for _, row in df.iterrows()]
    vec  = TfidfVectorizer(ngram_range=(1,2), max_features=8000, sublinear_tf=True)
    mat  = vec.fit_transform(docs)
    os.makedirs(os.path.dirname(IDX_PATH), exist_ok=True)
    with open(IDX_PATH,"wb") as f:
        pickle.dump({"vectorizer": vec, "matrix": mat, "documents": docs}, f)
    return vec, mat, docs, df

def build_rag_prompt(query: str, retrieved: list) -> str:
    context_parts = []
    for i, r in enumerate(retrieved, 1):
        d = r["data"]
        context_parts.append(
            f"Laptop {i}: {str(d['brand']).title()} — {str(d['Model'])[:60]}\n"
            f"  Price: {d['Price_SAR']} | Rating: {d['Rating']}/100 | Category: {d['Category']}\n"
            f"  CPU: {d['CPU_desc']} | RAM: {int(d['ram_memory'])}GB | Storage: {d['Storage']}\n"
            f"  GPU: {d['GPU_desc']} | Display: {d['Display']} | OS: {str(d['OS']).title()}\n"
            f"  Best For: {d['Best_For']} | Touch: {'Yes' if str(d['is_touch_screen']).lower()=='true' else 'No'}"
        )
    context = "\n\n".join(context_parts)
    return f"""You are EduLaptop Advisor, an AI assistant helping students in Saudi Arabia find the best laptop for education.

User asked: "{query}"

Top {len(retrieved)} laptops retrieved (prices in SAR):

{context}

Instructions:
- Answer ONLY based on the laptops above — no hallucination
- First output a markdown table with these columns: #, Brand & Model, Price (SAR), CPU, RAM, Storage, GPU, Rating
- Keep model names short (max 40 chars)
- After the table, write a "✅ Recommendation" section — max 2 sentences explaining the single best pick and why
- Do NOT write long paragraphs — table + short recommendation only

Answer:"""

def call_groq(prompt: str, api_key: str) -> str:
    client   = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model      = "llama-3.3-70b-versatile",
        messages   = [{"role": "user", "content": prompt}],
        max_tokens = 1000,
    )
    return response.choices[0].message.content

class LaptopRAG:

    def __init__(self, api_key: str | None = None):
        self.vectorizer, self.matrix, self.documents, self.df = build_index()
        self.api_key = api_key or os.getenv("GROQ_API_KEY")

    def extract_filters(self, query: str) -> dict:
        """Use Groq to extract structured filters from the user query."""
        if not self.api_key:
            return {}
        filter_prompt = f"""Extract laptop filter requirements from this query as JSON.
Query: "{query}"

Return ONLY a valid JSON object with these keys (use null if not mentioned):
{{
  "max_price_sar": <number or null>,
  "min_price_sar": <number or null>,
  "min_ram_gb": <number or null>,
  "gpu_type": <"dedicated" or "integrated" or null>,
  "gpu_brand": <"nvidia" or "amd" or "intel" or null>,
  "os": <"windows" or "mac" or "linux" or null>,
  "min_storage_gb": <number or null>,
  "processor_tier": <"core i3" or "core i5" or "core i7" or "core i9" or "ryzen 3" or "ryzen 5" or "ryzen 7" or null>,
  "max_rating": <number or null>,
  "min_rating": <number or null>
}}
Return ONLY the JSON, no explanation."""
        try:
            client   = Groq(api_key=self.api_key)
            response = client.chat.completions.create(
                model      = "llama-3.3-70b-versatile",
                messages   = [{"role": "user", "content": filter_prompt}],
                max_tokens = 200,
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            return json.loads(raw)
        except Exception:
            return {}

    def apply_filters(self, filters: dict):
        """Return filtered dataframe indices based on extracted filters."""
        mask = pd.Series([True] * len(self.df))

        if filters.get("max_price_sar"):
            mask &= self.df["Price_SAR_num"] <= float(filters["max_price_sar"])
        if filters.get("min_price_sar"):
            mask &= self.df["Price_SAR_num"] >= float(filters["min_price_sar"])
        if filters.get("min_ram_gb"):
            mask &= self.df["ram_memory"] >= int(filters["min_ram_gb"])
        if filters.get("gpu_type"):
            mask &= self.df["gpu_type"].str.lower() == filters["gpu_type"].lower()
        if filters.get("gpu_brand"):
            mask &= self.df["gpu_brand"].str.lower().str.contains(filters["gpu_brand"].lower())
        if filters.get("os"):
            os_map = {"mac": "mac", "linux": "ubuntu", "windows": "windows"}
            os_val = os_map.get(filters["os"].lower(), filters["os"].lower())
            mask &= self.df["OS"].str.lower().str.contains(os_val)
        if filters.get("min_storage_gb"):
            mask &= self.df["primary_storage_capacity"] >= int(filters["min_storage_gb"])
        if filters.get("processor_tier"):
            mask &= self.df["processor_tier"].str.lower().str.contains(
                filters["processor_tier"].lower().replace("core ","").replace("ryzen ","")
            )
        if filters.get("min_rating"):
            mask &= self.df["Rating"] >= int(filters["min_rating"])

        return mask

    def retrieve(self, query: str, top_k: int = 5):
        expanded = expand_query(query)
        q_vec    = self.vectorizer.transform([expanded])
        scores   = cosine_similarity(q_vec, self.matrix).flatten()

        # ── Smart filter via Groq ──────────────────────────────────────────
        filters      = self.extract_filters(query)
        filter_mask  = self.apply_filters(filters)
        valid_indices = self.df[filter_mask].index.tolist()

        if len(valid_indices) >= top_k:
            # Sort valid indices by score
            scored = sorted(valid_indices, key=lambda i: scores[i], reverse=True)
            candidates = scored[:top_k]
        else:
            # Fallback: no filter (too restrictive)
            candidates = scores.argsort()[::-1][:top_k]

        return [{"document": self.documents[i], "score": float(scores[i]),
                 "data": self.df.iloc[i].to_dict()} for i in candidates]

    def generate_answer(self, query: str, retrieved: list) -> str:
        if not retrieved or all(r["score"] == 0 for r in retrieved):
            return "No matching laptops found. Please try rephrasing your query."

        if self.api_key:
            try:
                return call_groq(build_rag_prompt(query, retrieved), self.api_key)
            except Exception:
                pass

        # Rule-based fallback — table format
        lines = [f"Here are the top {len(retrieved)} laptops matching your query:\n"]
        lines.append("| # | Brand & Model | Price (SAR) | CPU | RAM | Storage | GPU | Rating |")
        lines.append("|---|--------------|-------------|-----|-----|---------|-----|--------|")
        for i, r in enumerate(retrieved, 1):
            d     = r["data"]
            brand = str(d["brand"]).title()
            model = str(d["Model"])[:38]
            lines.append(
                f"| {i} | {brand} — {model} | {d['Price_SAR']} | {d['CPU_desc']} "
                f"| {int(d['ram_memory'])}GB | {d['Storage']} | {d['GPU_desc']} | {d['Rating']}/100 |"
            )
        # Short recommendation
        best  = retrieved[0]["data"]
        brand = str(best["brand"]).title()
        ql    = query.lower()
        if any(w in ql for w in ["budget","cheap","affordable","under","low"]):
            tip = "best value for budget students with solid SSD performance."
        elif any(w in ql for w in ["programming","coding","cs","engineering","software","ml"]):
            tip = "strong CPU and RAM make it ideal for coding and engineering workloads."
        elif any(w in ql for w in ["gaming","game"]):
            tip = "dedicated GPU delivers the best gaming and rendering performance."
        elif any(w in ql for w in ["light","portable","thin"]):
            tip = "lightweight build makes it the most portable option."
        elif any(w in ql for w in ["mac","apple","macbook"]):
            tip = "macOS ecosystem and battery life are unmatched for developers."
        else:
            tip = "it offers the best overall balance of specs and value."
        lines.append(f"\n✅ **Recommendation:** Go with the **{brand} — {str(best['Model'])[:45]}** ({best['Price_SAR']}) — {tip}")
        return "\n".join(lines)

    def answer(self, query: str, top_k: int = 5) -> dict:
        retrieved = self.retrieve(query, top_k=top_k)
        ans       = self.generate_answer(query, retrieved)
        return {"query": query, "answer": ans, "retrieved": retrieved}

    def get_stats(self) -> dict:
        return {
            "total":      len(self.df),
            "brands":     self.df["brand"].nunique(),
            "categories": self.df["Category"].nunique(),
            "price_min":  f"SAR {self.df['Price_SAR_num'].min():,.0f}",
            "price_max":  f"SAR {self.df['Price_SAR_num'].max():,.0f}",
        }