"""
File 4: 04_rag_vector_search_qa.py
==================================
GenAI & Deep Learning Assessment - Module 4
Topic: Retrieval-Augmented Generation (RAG) Document QA Pipeline

Description:
This module implements an end-to-end Retrieval-Augmented Generation (RAG) system.
It ingests unstructured text documents, splits them into semantic chunks with overlap,
computes dense vector embeddings, builds a searchable vector index, retrieves the top-k
most relevant context chunks for a user query, and formats a contextual prompt for an LLM.
"""

import numpy as np

class TextChunker:
    """Splits documents into overlapping chunks for indexing."""
    def __init__(self, chunk_size=150, chunk_overlap=30):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text):
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += (self.chunk_size - self.chunk_overlap)
        return chunks

class SimpleVectorStore:
    """Lightweight vector store utilizing TF-IDF / Embedding Cosine Similarity."""
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer()
        self.chunks = []
        self.vectors = None

    def add_texts(self, texts):
        self.chunks.extend(texts)
        self.vectors = self.vectorizer.fit_transform(self.chunks).toarray()

    def similarity_search(self, query, top_k=2):
        query_vec = self.vectorizer.transform([query]).toarray()
        # Compute cosine similarity
        norm_query = np.linalg.norm(query_vec)
        norm_vectors = np.linalg.norm(self.vectors, axis=1) + 1e-9
        
        similarities = np.dot(self.vectors, query_vec.T).squeeze() / (norm_vectors * norm_query)
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "chunk": self.chunks[idx],
                "score": float(similarities[idx])
            })
        return results

class RAGPipeline:
    """Orchestrates document retrieval and LLM context synthesis."""
    def __init__(self):
        self.chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        self.vector_store = SimpleVectorStore()

    def ingest_documents(self, documents):
        print(f"[1/3] Ingesting {len(documents)} document(s)...")
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.split_text(doc)
            all_chunks.extend(chunks)
        
        print(f"Created {len(all_chunks)} text chunks.")
        print("[2/3] Indexing chunks into Vector Store...")
        self.vector_store.add_texts(all_chunks)

    def generate_answer(self, query):
        print(f"\n[3/3] Processing Query: '{query}'")
        retrieved_docs = self.vector_store.similarity_search(query, top_k=2)

        print("\n--- Retrieved Relevant Context ---")
        context_str = ""
        for i, res in enumerate(retrieved_docs, 1):
            print(f"[{i}] (Relevance Score: {res['score']:.4f})")
            print(f"    {res['chunk']}\n")
            context_str += f"- {res['chunk']}\n"

        # Construct Prompt for LLM
        prompt = (
            f"You are a helpful AI assistant. Answer the user's question using ONLY the provided context.\n\n"
            f"Context:\n{context_str}\n"
            f"User Question: {query}\n\n"
            f"Answer:"
        )

        print("--- Generated Prompt for LLM Synthesis ---")
        print(prompt)
        
        # Synthetic LLM response demonstration
        simulated_answer = f"Based on the knowledge base, {retrieved_docs[0]['chunk'][:100]}..."
        return simulated_answer

def main():
    sample_knowledge_base = [
        """Generative AI is a category of artificial intelligence techniques capable of generating 
        new text, images, code, or audio based on learned patterns from training data. Transformer 
        architectures, introduced in 2017, form the backbone of modern Large Language Models (LLMs).""",
        
        """Retrieval-Augmented Generation (RAG) enhances LLM outputs by fetching authoritative 
        information from an external vector store before generating a response. RAG reduces hallucinations 
        and allows domain-specific data integration without expensive model fine-tuning.""",
        
        """Multi-Layer Perceptrons (MLPs) and Convolutional Neural Networks (CNNs) represent traditional 
        deep learning architectures for structured data and image computer vision tasks respectively."""
    ]

    pipeline = RAGPipeline()
    pipeline.ingest_documents(sample_knowledge_base)

    query = "What is RAG and how does it prevent hallucinations in Large Language Models?"
    answer = pipeline.generate_answer(query)
    
    print("\n==========================================")
    print("Final Synthesized Output:")
    print(answer)
    print("==========================================\n")

if __name__ == "__main__":
    main()
