# 🔹 Module 2 — Vector Databases in Practice

*Complete Day 2 Teaching Material*

---

## 1️⃣ Introduction to Vector Databases

### The Problem We're Solving

In Module 1, you learned to create embeddings.

But now you have a problem:

**Where do you store millions of embeddings?**

You cannot use:
- Regular files (too slow)
- Python lists (not persistent)
- Traditional databases (not optimized for vectors)

You need:

> **A database designed for high-dimensional vectors.**

---

### What Is a Vector Database?

A vector database is:

> A specialized database that stores, indexes, and searches vectors efficiently.

**Key capabilities:**
- Store embeddings with metadata
- Search by similarity (not keywords)
- Scale to millions of vectors
- Return results in milliseconds

---

### Why Not Use a Regular Database?

**Traditional databases (MySQL, PostgreSQL) are designed for:**
- Exact matches
- Sorting by value
- SQL queries

**Vector databases are designed for:**
- Approximate matches
- Similarity search
- High-dimensional data

**Example:**

Traditional database:
```sql
SELECT * FROM products WHERE name = 'laptop'
```

Vector database:
```python
results = collection.query(
    query_embeddings=[query_vector],
    n_results=5
)
```

The system automatically finds the closest matches.

---

### Real-World Use Cases

**Semantic Search:**
- Search documents by meaning, not keywords
- "Find articles about climate change" matches "global warming," "CO2 emissions," etc.

**Recommendation Systems:**
- "Find products similar to what this user liked"
- Netflix: "shows similar to this one"

**RAG Systems:**
- Retrieve relevant context before generating answers
- Chatbots that know your company docs

**Anomaly Detection:**
- Find unusual patterns in data
- Fraud detection, system monitoring

---

### Popular Vector Databases

| Database | Type | Best For |
|----------|------|----------|
| **Pinecone** | Cloud-native | Simplicity, managed service |
| **Weaviate** | Open-source | Feature-rich, GraphQL |
| **Qdrant** | Open-source | Performance, Rust-based |
| **ChromaDB** | Embedded/Server | Local development, Python-first |
| **pgvector** | PostgreSQL extension | Existing PostgreSQL users |

**We'll use ChromaDB because:**
- Zero configuration required
- Python-native
- Perfect for learning and prototyping
- Can scale to production
- Works locally and in-memory

---

## 2️⃣ ChromaDB Deep Dive

### What Is ChromaDB?

ChromaDB is:

> An open-source embedding database designed for AI applications.

It provides:
- Simple Python API
- Automatic persistence
- Built-in embedding functions
- Metadata filtering
- Multiple distance metrics

**Key philosophy:**

> "Make it easy to build AI applications with embeddings."

---

### ChromaDB Architecture

ChromaDB has three modes:

**1. In-Memory (Development)**
```python
import chromadb
client = chromadb.Client()
```
- Data lost when script ends
- Fast for testing

**2. Persistent (Local)**
```python
client = chromadb.PersistentClient(path="./chroma_db")
```
- Data saved to disk
- Survives restarts
- Good for development

**3. Client-Server (Production)**
```python
client = chromadb.HttpClient(host="localhost", port=8000)
```
- Run ChromaDB as a service
- Multiple clients can connect
- Production-ready

**For this course, we'll use Persistent mode.**

---

### Core Concepts

**Client:**
- Your connection to ChromaDB
- Manages all collections

**Collection:**
- Like a table in SQL
- Stores related embeddings
- Has a name and configuration

**Document:**
- A piece of text you want to store
- Automatically embedded (or provide your own)
- Can have metadata

**Metadata:**
- Additional information about documents
- Used for filtering
- Examples: author, date, category

---

### Distance Metrics in ChromaDB

ChromaDB supports multiple distance functions:

| Metric | Name | Best For |
|--------|------|----------|
| `cosine` | Cosine Similarity | Semantic search (default) |
| `l2` | Euclidean Distance | Spatial data |
| `ip` | Inner Product | Pre-normalized vectors |

**For semantic search, use cosine (the default).**

Cosine measures the angle between vectors, not their length.

---

### How ChromaDB Indexes Work

**Without an index:**
- Compare query to every vector
- Slow for large datasets
- O(n) time complexity

**With an index (HNSW):**
- Hierarchical Navigable Small World graph
- Approximate nearest neighbor search
- Much faster
- O(log n) time complexity

**ChromaDB automatically creates indexes** when your collection grows.

---

## 3️⃣ Setting Up Your Vector Database

### Installation

```bash
pip install chromadb
```

That's it. No server setup required.

---

### Creating Your First Collection

```python
import chromadb

# Create persistent client
client = chromadb.PersistentClient(path="./chroma_db")

# Create a collection
collection = client.create_collection(
    name="documents",
    metadata={"description": "My document collection"}
)

print("Collection created:", collection.name)
```

**What just happened:**
- ChromaDB created a folder `./chroma_db`
- Created a collection named "documents"
- Ready to store embeddings

---

### Collection Configuration

You can configure collections with specific settings:

```python
collection = client.create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}  # Distance metric
)
```

**Common configurations:**

```python
# L2 distance
metadata={"hnsw:space": "l2"}

# Inner product
metadata={"hnsw:space": "ip"}

# Cosine (default)
metadata={"hnsw:space": "cosine"}
```

---

### Getting or Creating Collections

**Problem:** Creating a collection twice causes an error.

**Solution:** Use `get_or_create_collection`

```python
collection = client.get_or_create_collection(
    name="documents"
)
```

This is safer for production code.

---

### Listing Collections

```python
# Get all collections
collections = client.list_collections()

for col in collections:
    print(f"Collection: {col.name}")
```

---

### Deleting Collections

```python
# Delete a collection
client.delete_collection(name="documents")
```

**Warning:** This permanently deletes all data.

---

## 4️⃣ Storing Embeddings in ChromaDB

### The Basic Pattern

There are two ways to store data:

**1. Let ChromaDB create embeddings (automatic)**
**2. Provide your own embeddings (manual)**

We'll use **manual mode** so we have full control.

---

### Manual Embedding Workflow

```python
from sentence_transformers import SentenceTransformer
import chromadb

# Load embedding model
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Create client and collection
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="documents")

# Prepare data
documents = [
    "Python is a programming language",
    "Machine learning is a subset of AI",
    "ChromaDB stores vector embeddings"
]

# Generate embeddings
embeddings = model.encode(documents).tolist()

# Store in ChromaDB
collection.add(
    ids=["doc1", "doc2", "doc3"],
    embeddings=embeddings,
    documents=documents
)

print("Added 3 documents to collection")
```

---

### Understanding the `add` Method

```python
collection.add(
    ids=["doc1", "doc2", "doc3"],        # Unique identifiers
    embeddings=embeddings,                # Vector representations
    documents=documents,                  # Original text (optional)
    metadatas=[{"type": "tech"}] * 3     # Additional info (optional)
)
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `ids` | List[str] | Yes | Unique ID for each document |
| `embeddings` | List[List[float]] | Yes* | Vector representations |
| `documents` | List[str] | No | Original text |
| `metadatas` | List[dict] | No | Additional information |

*Not required if using automatic embedding.

---

### Adding Metadata

Metadata allows filtering:

```python
collection.add(
    ids=["doc1", "doc2", "doc3"],
    embeddings=embeddings,
    documents=documents,
    metadatas=[
        {"category": "programming", "year": 2024},
        {"category": "AI", "year": 2024},
        {"category": "database", "year": 2024}
    ]
)
```

Now you can search within specific categories.

---

### Handling Large Batches

For large datasets, add in batches:

```python
batch_size = 100

for i in range(0, len(documents), batch_size):
    batch_docs = documents[i:i+batch_size]
    batch_embeddings = model.encode(batch_docs).tolist()
    batch_ids = [f"doc{j}" for j in range(i, i+len(batch_docs))]
    
    collection.add(
        ids=batch_ids,
        embeddings=batch_embeddings,
        documents=batch_docs
    )
    
    print(f"Added batch {i//batch_size + 1}")
```

---

### Updating Documents

```python
collection.update(
    ids=["doc1"],
    embeddings=[new_embedding],
    documents=["Updated content"],
    metadatas=[{"category": "updated"}]
)
```

---

### Deleting Documents

```python
# Delete by ID
collection.delete(ids=["doc1", "doc2"])

# Delete by filter
collection.delete(where={"category": "old"})
```

---

### Getting Collection Statistics

```python
# Count documents
count = collection.count()
print(f"Total documents: {count}")

# Peek at first few documents
results = collection.peek(limit=5)
print(results)
```

---

## 5️⃣ Handling Dependency Issues

### Common Installation Problems

**Problem 1: Version conflicts**

```bash
ERROR: pip's dependency resolver does not currently take into account all the packages
```

**Solution:**
```bash
pip install --upgrade pip
pip install chromadb --no-cache-dir
```

---

**Problem 2: Missing system dependencies (Linux)**

```bash
ERROR: Failed building wheel for hnswlib
```

**Solution:**
```bash
sudo apt-get update
sudo apt-get install build-essential python3-dev
pip install chromadb
```

---

**Problem 3: Apple Silicon (M1/M2) issues**

```bash
ERROR: Could not build wheels for hnswlib
```

**Solution:**
```bash
brew install cmake
pip install chromadb
```

---

### Virtual Environment Best Practices

Always use a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install chromadb sentence-transformers
```

---

### Verifying Installation

```python
import chromadb
print(f"ChromaDB version: {chromadb.__version__}")

import sentence_transformers
print(f"SentenceTransformers version: {sentence_transformers.__version__}")
```

---

### Managing Dependencies with requirements.txt

Create `requirements.txt`:

```
chromadb==0.4.22
sentence-transformers==2.3.1
numpy==1.24.3
python-dotenv==1.0.0
```

Install:

```bash
pip install -r requirements.txt
```

---

## 6️⃣ Complete Working Example

### Building a Knowledge Base

```python
from sentence_transformers import SentenceTransformer
import chromadb

# Initialize
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
client = chromadb.PersistentClient(path="./knowledge_base")
collection = client.get_or_create_collection(name="tech_docs")

# Sample documents
documents = [
    "Python is a high-level programming language known for its simplicity",
    "Machine learning algorithms learn patterns from data",
    "Neural networks are inspired by biological neurons",
    "ChromaDB is a vector database for AI applications",
    "Embeddings convert text into numerical representations"
]

# Generate embeddings
print("Generating embeddings...")
embeddings = model.encode(documents).tolist()

# Store in ChromaDB
print("Storing in database...")
collection.add(
    ids=[f"doc{i}" for i in range(len(documents))],
    embeddings=embeddings,
    documents=documents,
    metadatas=[{"source": "tech_guide", "index": i} for i in range(len(documents))]
)

print(f"Successfully stored {len(documents)} documents")
print(f"Collection size: {collection.count()}")
```

---

### Querying the Knowledge Base

```python
# Query
query = "What is a vector database?"

# Generate query embedding
query_embedding = model.encode([query]).tolist()

# Search
results = collection.query(
    query_embeddings=query_embedding,
    n_results=2
)

# Display results
print(f"\nQuery: {query}\n")
print("Top results:")
for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
    print(f"\n{i+1}. {doc}")
    print(f"   Distance: {distance:.4f}")
```

---

### Adding Metadata Filtering

```python
# Query with filter
results = collection.query(
    query_embeddings=query_embedding,
    n_results=2,
    where={"source": "tech_guide"}
)
```

---

### Complete Production-Ready Script

```python
import os
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict

class VectorStore:
    def __init__(self, collection_name: str, persist_directory: str = "./chroma_db"):
        """Initialize vector store with ChromaDB"""
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None):
        """Add documents to the vector store"""
        embeddings = self.model.encode(documents).tolist()
        ids = [f"doc{i}" for i in range(self.collection.count(), self.collection.count() + len(documents))]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        return ids
    
    def search(self, query: str, n_results: int = 5, filter_criteria: Dict = None):
        """Search for similar documents"""
        query_embedding = self.model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=filter_criteria
        )
        
        return results
    
    def get_count(self):
        """Get total number of documents"""
        return self.collection.count()

# Usage
if __name__ == "__main__":
    # Initialize
    store = VectorStore("my_documents")
    
    # Add documents
    docs = [
        "Python is great for data science",
        "Machine learning requires large datasets",
        "ChromaDB simplifies vector storage"
    ]
    store.add_documents(docs)
    
    # Search
    results = store.search("How do I store vectors?", n_results=2)
    
    print("Search Results:")
    for doc in results['documents'][0]:
        print(f"- {doc}")
```

---

## 🎯 Module 2 Outcome

After this module, students:

✅ Understand what vector databases are and why they're needed  
✅ Know how ChromaDB works internally  
✅ Can create and configure collections  
✅ Can store embeddings with metadata  
✅ Can handle common dependency issues  
✅ Have built a complete vector storage system  

**They now have the infrastructure required for:**

👉 Semantic search systems  
👉 Document retrieval  
👉 RAG applications  

---

## 📝 Practice Exercises

### Exercise 1: Build a Quote Database

Create a vector store for famous quotes:
- Store 20 quotes with author metadata
- Search by theme or concept
- Filter by author

### Exercise 2: Personal Note System

Build a system that:
- Stores your daily notes
- Adds timestamps as metadata
- Searches by semantic meaning
- Filters by date range

### Exercise 3: Error Handling

Enhance the VectorStore class with:
- Try-except blocks
- Validation for empty inputs
- Duplicate ID detection
- Connection error handling

---

This completes Module 2.
