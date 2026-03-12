# 🔍 Semantic Search & RAG Pipeline — Flipkart Product Reviews

> **Built by:** Bonamukkala Charan Reddy  
> **Institution:** NxtWave Institute | BITS Pilani (Online Degree)  
> **Project Type:** AI/ML — Vector Search, NLP, RAG Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bonamukkala-bot/semantic-search-rag-flipkart/blob/main/semantic_search_system%20(5).ipynb)
[![GitHub](https://img.shields.io/badge/GitHub-bonamukkala--bot-blue?logo=github)](https://github.com/bonamukkala-bot)

---

## 📌 Project Overview

Traditional keyword-based search systems fail when users express their intent
using different words than those stored in the database.

**Example Problem:**
- User searches → `"energy efficient AC"`
- Database has → `"low power air conditioner"`
- Keyword search → ❌ No results found
- Our Semantic Search → ✅ Finds it instantly!

This project builds a **complete Semantic Search and RAG pipeline** that
understands user intent and retrieves results based on **meaning**, not just
keyword matching.

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Name** | Flipkart Product Customer Reviews |
| **Total Reviews** | 205,052 |
| **Columns** | product_name, product_price, Rate, Review, Summary, Sentiment |
| **Domain** | E-commerce Product Reviews |
| **Sentiments** | Positive, Negative, Neutral |

---

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Pandas** | Data loading and cleaning |
| **Matplotlib & Seaborn** | Data visualizations |
| **WordCloud** | Word frequency visualization |
| **Sentence-Transformers (SBERT)** | Generate 384-dim text embeddings |
| **FAISS (IndexFlatIP)** | Fast vector similarity search |
| **Groq LLaMA 3** | AI answer generation (RAG) |
| **Scikit-learn PCA** | 2D embedding visualization |

---

## 🚀 Pipeline Architecture
```
User Query
    ↓
Convert to Vector (SBERT - all-MiniLM-L6-v2)
    ↓
Normalize with FAISS L2
    ↓
Search FAISS Index (cosine similarity)
    ↓
Retrieve Top-K Relevant Reviews
    ↓
Send to Groq LLaMA 3 as Context
    ↓
Generate Smart Natural Language Answer ✅
```

---

## 📁 Project Structure
```
semantic-search-rag-flipkart/
├── 📓 semantic_search_system (5).ipynb   ← Main notebook
├── 📊 eda_plots (5).png                  ← EDA visualizations
├── ☁️  wordcloud (3).png                 ← Word cloud
├── 🧠 pca_embeddings (3).png             ← PCA embedding plot
├── 📈 evaluation_metrics (4).png         ← Precision & Recall charts
└── 📝 business_report (4).txt            ← Business insights report
```

---

## 💻 Complete Code — Cell by Cell

---

### 🟦 CELL 1 — Install Libraries
```python
# Install all required libraries
!pip install sentence-transformers faiss-cpu wordcloud matplotlib seaborn pandas numpy scikit-learn transformers torch --quiet

print("✅ All libraries installed successfully!")
```

---

### 🟦 CELL 2 — Load Dataset
```python
# Load the already-uploaded dataset
import pandas as pd

df = pd.read_csv('/content/Dataset-SA.csv', encoding='latin-1', on_bad_lines='skip')

print(f"✅ Dataset loaded successfully!")
print(f"📊 Shape: {df.shape}")
print(f"\n📋 Columns: {list(df.columns)}")
df.head(3)
```

---

### 🟦 CELL 3 — Explore Dataset
```python
# Explore what's inside the dataset (EDA)
print("=" * 60)
print("📊 DATASET OVERVIEW")
print("=" * 60)

print(f"\n🔢 Total rows: {df.shape[0]}")
print(f"🔢 Total columns: {df.shape[1]}")
print(f"\n📋 Columns: {list(df.columns)}")
print(f"\n❓ Missing values:")
print(df.isnull().sum())
print(f"\n⭐ Rating distribution:")
print(df['Rate'].value_counts().sort_index())
print(f"\n😊 Sentiment distribution:")
print(df['Sentiment'].value_counts())
df.head(5)
```

---

### 🟦 CELL 4 — Rename Columns
```python
# Rename columns to standard names
df = df.rename(columns={
    'product_name': 'product_name',
    'Rate': 'rating',
    'Review': 'review_text',
    'Summary': 'summary',
    'Sentiment': 'sentiment',
    'product_price': 'product_price'
})

print("✅ Columns renamed!")
print("📋 Final columns:", list(df.columns))
df.head(3)
```

---

### 🟦 CELL 5 — Clean & Preprocess Data
```python
# Clean and preprocess the text data
import re

print("🧹 Cleaning data...")

df = df.dropna(subset=['review_text'])
df['review_text'] = df['review_text'].astype(str)
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df = df.dropna(subset=['rating'])
df['rating'] = df['rating'].astype(float)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleaned_review'] = df['review_text'].apply(clean_text)
df['text_length'] = df['cleaned_review'].apply(len)
df = df[df['text_length'] > 10].reset_index(drop=True)

if len(df) > 5000:
    df = df.sample(5000, random_state=42).reset_index(drop=True)
    print(f"⚡ Sampled 5000 rows for speed")

print(f"✅ Cleaned! Final size: {df.shape}")
print(df['sentiment'].value_counts())
```

---

### 🟦 CELL 6 — EDA Visualizations
```python
# EDA - Rating, Text Length & Sentiment Distribution
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("📊 Flipkart Reviews - EDA Analysis", fontsize=16, fontweight='bold')

# Rating Distribution
rating_counts = df['rating'].value_counts().sort_index()
colors = ['#FF4444','#FF8800','#FFCC00','#88CC00','#00AA00']
bars = axes[0].bar(rating_counts.index, rating_counts.values,
                   color=colors[:len(rating_counts)], edgecolor='white')
axes[0].set_title('⭐ Rating Distribution', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Rating')
axes[0].set_ylabel('Number of Reviews')

# Text Length Distribution
axes[1].hist(df['text_length'], bins=40, color='#4A90D9', edgecolor='white')
axes[1].set_title('📝 Review Length Distribution', fontsize=13, fontweight='bold')
axes[1].axvline(df['text_length'].mean(), color='red', linestyle='--',
                label=f'Mean: {df["text_length"].mean():.0f}')
axes[1].legend()

# Sentiment Distribution
sentiment_counts = df['sentiment'].value_counts()
axes[2].bar(sentiment_counts.index, sentiment_counts.values,
            color=['#00AA00','#FF4444','#FFCC00'], edgecolor='white')
axes[2].set_title('😊 Sentiment Distribution', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('eda_plots.png', bbox_inches='tight')
plt.show()
```

### 📊 EDA Output:
![EDA Plots](eda_plots%20(5).png)

---

### 🟦 CELL 7 — Word Cloud
```python
# Word Cloud - Most Common Words in Reviews
from wordcloud import WordCloud

stopwords = set(['the','a','an','and','or','but','is','are','was','were',
                 'i','me','my','we','you','your','they','them','very',
                 'just','also','product','item','buy','bought'])

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# All reviews
all_text = ' '.join(df['cleaned_review'].tolist())
wc_all = WordCloud(width=800, height=400, background_color='white',
                   colormap='viridis', stopwords=stopwords, max_words=150)
wc_all.generate(all_text)
axes[0].imshow(wc_all, interpolation='bilinear')
axes[0].axis('off')
axes[0].set_title('All Reviews', fontsize=13, fontweight='bold')

# Positive reviews only
pos_text = ' '.join(df[df['sentiment']=='positive']['cleaned_review'].tolist())
wc_pos = WordCloud(width=800, height=400, background_color='#f0fff0',
                   colormap='Greens', stopwords=stopwords, max_words=100)
wc_pos.generate(pos_text)
axes[1].imshow(wc_pos, interpolation='bilinear')
axes[1].axis('off')
axes[1].set_title('Positive Reviews Only', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('wordcloud.png', bbox_inches='tight')
plt.show()
```

### ☁️ Word Cloud Output:
![Word Cloud](wordcloud%20(3).png)

---

### 🟦 CELL 8 — Generate Sentence Embeddings
```python
# Generate Embeddings using Sentence-Transformers
from sentence_transformers import SentenceTransformer
import numpy as np

print("🤖 Loading Sentence-Transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model loaded!")

print(f"\n⚡ Generating embeddings for {len(df)} reviews...")
embeddings = model.encode(
    df['cleaned_review'].tolist(),
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)

print(f"✅ Embeddings done!")
print(f"📐 Shape: {embeddings.shape}")
# Output: (5000, 384) → 5000 reviews × 384 dimensions
```

**What this does:**
- Converts each review into a **384-dimensional vector**
- Similar meaning reviews get **similar vectors**
- Example: `"great product"` and `"excellent item"` → very similar vectors

---

### 🟦 CELL 9 — Build FAISS Index
```python
# Build FAISS Vector Index (Correct Order!)
import faiss

print("🏗️ Building FAISS Index...")

dimension = embeddings.shape[1]  # 384

# ✅ CORRECT ORDER: Normalize FIRST, then add
embeddings_normalized = np.array(embeddings).astype('float32').copy()
faiss.normalize_L2(embeddings_normalized)   # Step 1: Normalize

index = faiss.IndexFlatIP(dimension)        # Step 2: Create index
index.add(embeddings_normalized)            # Step 3: Add

embeddings_f32 = embeddings_normalized

print(f"✅ FAISS Index built!")
print(f"📊 Total vectors indexed: {index.ntotal}")
```

---

### 🟦 CELL 10 — Semantic Search Function
```python
# Semantic Search - finds reviews by MEANING not keywords!
def semantic_search(query, top_k=5, sentiment_filter=None):
    print(f"\n🔍 Searching: '{query}'")
    print("-" * 55)

    # Encode query
    q_emb = model.encode([query], normalize_embeddings=True).astype('float32')

    # Search FAISS
    scores, indices = index.search(q_emb, top_k * 4)

    # Build results
    results = []
    for score, idx in zip(scores[0], indices[0]):
        row = df.iloc[idx].to_dict()
        row['similarity_score'] = float(score)
        results.append(row)

    results_df = pd.DataFrame(results)

    if sentiment_filter:
        results_df = results_df[results_df['sentiment'] == sentiment_filter]

    results_df = results_df.head(top_k).reset_index(drop=True)

    for i, row in results_df.iterrows():
        stars = '⭐' * int(row['rating'])
        print(f"\n📌 Result #{i+1}")
        print(f"   🎯 Similarity: {row['similarity_score']:.4f}")
        print(f"   {stars} Rating: {row['rating']}/5")
        print(f"   💬 Review: {str(row['review_text'])[:180]}...")

    return results_df

# Test queries
r1 = semantic_search("energy efficient cooling appliance", top_k=3)
r2 = semantic_search("product stopped working defective", top_k=3)
r3 = semantic_search("fast delivery excellent packaging", top_k=3)
```

---

### 🟦 CELL 12 — PCA Visualization
```python
# PCA - Visualize Embeddings in 2D
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
emb_2d = pca.fit_transform(embeddings_f32[:500])

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Colored by rating
scatter = axes[0].scatter(emb_2d[:, 0], emb_2d[:, 1],
                          c=df['rating'][:500], cmap='RdYlGn', alpha=0.6, s=20)
plt.colorbar(scatter, ax=axes[0], label='Rating')
axes[0].set_title('Reviews by Rating\n🟢 Positive  🔴 Negative')

# Colored by sentiment
color_map = {'positive':'#00AA00','negative':'#FF4444','neutral':'#FFAA00'}
colors_list = [color_map.get(s,'#888') for s in df['sentiment'][:500]]
axes[1].scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors_list, alpha=0.6, s=20)
axes[1].set_title('Reviews by Sentiment')

plt.tight_layout()
plt.savefig('pca_embeddings.png', bbox_inches='tight')
plt.show()
```

### 🧠 PCA Embedding Output:
![PCA Embeddings](pca_embeddings%20(3).png)

---

### 🟦 CELL 13 — RAG Pipeline
```python
# RAG Pipeline using Groq LLaMA
from groq import Groq
from google.colab import userdata

client = Groq(api_key=userdata.get('GROQ_API_KEY'))

def rag_pipeline(question, top_k=5):
    print(f"\n❓ QUESTION: {question}")

    # Step 1: Retrieve
    q_emb = model.encode([question], normalize_embeddings=True).astype('float32')
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, top_k * 5)

    # Deduplicate
    seen, unique = set(), []
    for score, idx in zip(scores[0], indices[0]):
        text = str(df.iloc[idx]['review_text'])[:100]
        if text not in seen:
            seen.add(text)
            unique.append((score, idx))
        if len(unique) == top_k:
            break

    # Build context
    context = "\n".join([
        f"R{i+1}(⭐{df.iloc[idx]['rating']},{df.iloc[idx]['sentiment']}): "
        f"{str(df.iloc[idx]['review_text'])[:80]}"
        for i, (score, idx) in enumerate(unique)
    ])

    # Step 2: Generate
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content":
            f"Based on these reviews, answer briefly.\n\n"
            f"Reviews:\n{context}\n\nQuestion: {question}\nAnswer:"}],
        max_tokens=150,
        temperature=0.3
    )

    answer = response.choices[0].message.content.strip()
    print(f"\n💡 AI ANSWER: {answer}")
    return answer

# Tests
rag_pipeline("Is this product good for long term use?")
rag_pipeline("What do customers say about delivery?")
rag_pipeline("What are the most common complaints?")
```

---

### 🟦 CELL 15 — Evaluation Metrics
```python
# Precision@K and Recall@K using keyword-based ground truth
import numpy as np

def get_ground_truth(query, df, top_n=50):
    keywords = query.lower().split()
    relevant = []
    for i, row in df.iterrows():
        text = str(row['cleaned_review']).lower()
        if any(kw in text for kw in keywords if len(kw) > 3):
            relevant.append(i)
    return set(relevant[:top_n])

def evaluate_semantic_search(test_queries, k_values=[1, 3, 5, 10]):
    all_precision = {k: [] for k in k_values}
    all_recall    = {k: [] for k in k_values}

    for query in test_queries:
        relevant_set = get_ground_truth(query, df)
        if not relevant_set:
            continue

        q_emb = model.encode([query], normalize_embeddings=True).astype('float32')
        faiss.normalize_L2(q_emb)
        _, indices = index.search(q_emb, max(k_values))
        retrieved = list(indices[0])

        for k in k_values:
            top_k = set(retrieved[:k])
            relevant_in_k = len(top_k & relevant_set)
            all_precision[k].append(relevant_in_k / k)
            all_recall[k].append(relevant_in_k / len(relevant_set))

    return all_precision, all_recall

test_queries = [
    "quality good excellent product recommended",
    "delivery fast packaging safe arrived",
    "broken defective stopped working returned",
    "value money affordable budget cheap",
    "durable long lasting build quality strong",
    "customer service support helpful response",
    "battery life performance speed",
    "size fitting comfortable design look",
]

all_precision, all_recall = evaluate_semantic_search(test_queries)

for k in [1, 3, 5, 10]:
    print(f"K={k} | Precision@{k}: {np.mean(all_precision[k]):.4f} | "
          f"Recall@{k}: {np.mean(all_recall[k]):.4f}")
```

### 📈 Evaluation Metrics Output:
![Evaluation Metrics](evaluation_metrics%20(4).png)

---

## 📊 Results Summary

| Metric | K=1 | K=3 | K=5 | K=10 |
|--------|-----|-----|-----|------|
| **Precision@K** | High | Medium | Medium | Lower |
| **Recall@K** | Low | Medium | Higher | Highest |

> Higher K = More results = Better Recall but Lower Precision

---

## 💼 Business Insights

### Key Findings:
- **80% Positive** reviews (4002/5000 sampled)
- **15% Negative** reviews (773/5000) — contain critical feedback
- **5% Neutral** reviews (225/5000)

### 5 Business Recommendations:

| # | Recommendation | Impact |
|---|---------------|--------|
| 1 | Replace keyword search with semantic search | 30-40% better results |
| 2 | Auto-generate review summaries using RAG | Faster buying decisions |
| 3 | Detect recurring complaints automatically | Better seller alerts |
| 4 | Cross-sell using embedding similarity | Higher conversion rate |
| 5 | Extend to Hindi/regional languages | Wider customer reach |

---

## 🔑 Key Concepts Used

| Concept | Explanation |
|---------|------------|
| **Text Embeddings** | Convert text to 384-dim vectors capturing meaning |
| **Cosine Similarity** | Measure how similar two vectors are |
| **FAISS IndexFlatIP** | Search millions of vectors in milliseconds |
| **RAG** | Retrieve relevant docs → Generate AI answer |
| **Precision@K** | What % of top-K results are relevant |
| **Recall@K** | What % of all relevant docs found in top-K |

---

## 👨‍💻 About the Author

**Bonamukkala Charan Reddy**
- 🎓 BSc Computer Science (AI/ML) — NxtWave Institute
- 📚 Online Degree — BITS Pilani
- 💼 [Portfolio](https://charan-me.vercel.app)
- 🐙 [GitHub](https://github.com/bonamukkala-bot)
- 📸 [Instagram](https://instagram.com/trending.tech.ai)
