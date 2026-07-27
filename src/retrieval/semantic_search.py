import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_vector_database():
    index = faiss.read_index("data/vector_store/faiss_index.index")

    with open("data/vector_store/chunks.pkl", "rb") as file:
        chunks = pickle.load(file)

    return index, chunks


def search_documents(query, top_k=3):

    model = SentenceTransformer("all-MiniLM-L6-v2")

    index, chunks = load_vector_database()

    # Convert the question into an embedding
    query_embedding = model.encode([query])

    query_embedding = np.array(query_embedding).astype("float32")

    # Search the vector database
    distances, indices = index.search(query_embedding, top_k)

    results = []

    for i in indices[0]:
        results.append(chunks[i])

    return results


if __name__ == "__main__":

    question = input("Enter your question: ")

    results = search_documents(question)

    print("\nTop Matching Chunks:\n")

    for number, result in enumerate(results, start=1):

        print(f"Result {number}")
        print(f"Document: {result['file_name']}")
        print(result["text"][:500])
        print("-" * 80)