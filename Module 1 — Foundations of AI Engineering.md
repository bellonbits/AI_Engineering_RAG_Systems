# 🔹 Module 1 — Foundations of AI Engineering

*Complete Day 1 Teaching Material*

---

## 1️⃣ Your Next Big Step in AI Engineering

### What Is AI Engineering?

AI Engineering is:

> **Building real systems using AI models.**

**It is NOT:**
- Training massive models from scratch
- Doing AI research
- Writing mathematical proofs

**It IS:**
- Using pretrained models
- Designing data pipelines
- Connecting models to real-world data
- Solving real problems

---

### How Modern AI Systems Work

Every modern AI system has 3 layers:

1. **Model Layer**
   - LLMs (generate text)
   - Embedding models (convert text to vectors)

2. **Data Layer**
   - Your documents
   - Your company knowledge
   - Your database

3. **Retrieval Layer**
   - Search
   - Similarity
   - Ranking

**Today, we focus on the foundation:**
👉 Embeddings and similarity.

---

### Why This Step Matters

Large language models do NOT:
- Automatically know your private data
- Automatically search your documents
- Automatically remember past conversations

**To solve this, we convert text into numbers.**

That is the foundation of AI Engineering.

---

## 2️⃣ What Are Embeddings?

### Simple Definition

An embedding is:

> **A numerical representation of meaning.**

When you give a sentence to an embedding model, it returns a list of numbers.

**Example:**

```
"I love dogs"
→ [0.21, -0.54, 0.67, ...]
```

These numbers represent the **meaning** of the sentence.

---

### Why Convert Text to Numbers?

**Computers understand:**
- Numbers
- Math
- Geometry

**They do NOT understand:**
- Emotion
- Context
- Language directly

**So we convert:**

```
Text → Vector → Geometry → Similarity
```

---

### What Is a Vector?

A vector is simply:

> **A list of numbers.**

**Example:**

```
[3, 5] → 2 dimensions
[1, 7, 2] → 3 dimensions
```

**Our embedding model produces:**
- 384 numbers
- So it's a 384-dimensional vector

---

### What Does "384-Dimensional" Mean?

Imagine:

**In 2D space:**
```
(x, y)
```

**In 3D space:**
```
(x, y, z)
```

**Embeddings use:**
```
(x1, x2, x3, ..., x384)
```

Each sentence becomes a **point in a 384-dimensional space**.

You cannot visualize 384D — but mathematically, it works the same way as 2D.

---

### What Does Each Dimension Mean?

**Important:**

- Dimension 1 does NOT mean "animals"
- Dimension 2 does NOT mean "positive emotion"

Each dimension captures **part of complex patterns**.

Meaning exists **across the entire vector** — not inside one number.

**Think of it like:**

The human brain — meaning is in patterns of neurons, not one single neuron.

---

### Dense Representation

Embeddings are **dense**:

```
[0.21, -0.54, 0.67, 0.11, ...]
```

Most numbers are non-zero.

This is different from keyword-based systems, which are **sparse**:

```
[0, 0, 1, 0, 0, 0, 1, 0]
```

- **Dense vectors** capture meaning
- **Sparse vectors** capture word presence

---

## 3️⃣ How Vector Similarity Works

Now we ask: If embeddings are vectors, how do we compare them?

---

### The Core Idea

**If two vectors point in similar directions, they have similar meaning.**

We measure this using:

👉 **Cosine Similarity**

---

### Why Direction Matters

Consider two vectors:

```
A = (2, 2)
B = (4, 4)
```

They point in the **same direction**. But B is longer.

They represent the same direction (same meaning).

**Now:**

```
C = (-2, 2)
```

Completely different direction.

**Meaning is about direction, not length.**

---

### Cosine Similarity Formula

$$\text{similarity} = \frac{A \cdot B}{||A|| \times ||B||}$$

**Where:**
- $A \cdot B$ = dot product
- $||A||$ = magnitude (length)
- Result between -1 and 1

---

### What Do the Scores Mean?

| Score   | Meaning          |
|---------|------------------|
| 0.9+    | Very similar     |
| 0.7–0.9 | Related          |
| 0.4–0.7 | Somewhat related |
| < 0.4   | Mostly unrelated |

**In real AI systems:**
0.7+ is usually considered relevant.

---

### Real Example

```python
from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

text1 = "I love dogs"
text2 = "I adore puppies"
text3 = "Python programming tutorial"

emb1 = model.encode(text1)
emb2 = model.encode(text2)
emb3 = model.encode(text3)

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

print(cosine_similarity(emb1, emb2))  # High
print(cosine_similarity(emb1, emb3))  # Low
```

**Students will see:**
- Similar sentences → high score
- Unrelated sentences → low score

This is **semantic similarity**.

---

## 4️⃣ Setting Up Environment Variables

Good engineering starts with discipline.

**Never hardcode API keys.**

---

### Step 1 — Install Dependencies

```bash
pip install sentence-transformers groq python-dotenv numpy
```

---

### Step 2 — Create `.env` File

```
GROQ_API_KEY=your_api_key_here
```

---

### Step 3 — Load Environment Variables

```python
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
print(api_key)
```

**This ensures:**
- Security
- Clean architecture
- Production readiness

---

## 5️⃣ Creating Your First Embedding

Now we build.

---

### Load the Model

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
```

The first run downloads the model. After that, it works offline.

---

### Create an Embedding

```python
text = "AI engineering is powerful"

embedding = model.encode(text)

print(type(embedding))   # numpy array
print(len(embedding))    # 384
```

**Now the sentence has become math.**

---

### Compare Two Sentences

```python
text1 = "Machine learning is exciting"
text2 = "AI is fascinating"

emb1 = model.encode(text1)
emb2 = model.encode(text2)

similarity = dot(emb1, emb2) / (norm(emb1) * norm(emb2))

print("Similarity:", similarity)
```

**Students now see:**

```
Language → Vector → Geometry → Meaning comparison
```
