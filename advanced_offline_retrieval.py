# advanced_offline_retrieval.py
"""
Advanced offline retrieval module for Kelly AI.
Uses FAISS for vector-based retrieval of offline knowledge.
"""
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import database
from logger_config import setup_logger

logger = setup_logger("advanced_offline_retrieval", "./logs/advanced_offline_retrieval.log")

def build_faiss_index(texts):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)
    vectors = tfidf_matrix.toarray().astype("float32")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index, vectorizer, vectors

def retrieve_offline_knowledge_faiss(query, top_k=5):
    try:
        items = database.get_relevant_knowledge("", limit=1000, only_new=False)
        if not items:
            logger.info("No offline knowledge found.")
            return []
        texts = [item.content for item in items]
        ids = [item.id for item in items]
        index, vectorizer, _ = build_faiss_index(texts)
        query_vec = vectorizer.transform([query]).toarray().astype("float32")
        distances, indices = index.search(query_vec, top_k)
        retrieved = []
        for idx, dist in zip(indices[0], distances[0]):
            retrieved.append({"id": ids[idx], "content": texts[idx], "similarity": 1.0 / (1.0 + float(dist))})
        logger.info(f"FAISS retrieved {len(retrieved)} items for query '{query}'.")
        return retrieved
    except Exception as e:
        logger.error(f"Error in advanced offline retrieval: {e}")
        return []