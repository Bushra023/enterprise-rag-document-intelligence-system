import os
import pickle

import faiss
import numpy as np

from src.embeddings.embedding_generator import generate_embeddings


def build_vector_database(folder_path):

    chunks, embeddings = generate_embeddings(folder_path)

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    os.makedirs("data/vector_store", exist_ok=True)

    faiss.write_index(
        index,
        "data/vector_store/faiss_index.index"
    )

    with open(
        "data/vector_store/chunks.pkl",
        "wb"
    ) as file:

        pickle.dump(chunks, file)

    return index, chunks


if __name__ == "__main__":

    index, chunks = build_vector_database("data/raw")

    print("\nVector Database Created Successfully!")

    print(f"Total Vectors Stored: {index.ntotal}")

    print(f"Vector Dimension: {index.d}")