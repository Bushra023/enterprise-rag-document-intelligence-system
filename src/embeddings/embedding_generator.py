from sentence_transformers import SentenceTransformer
from src.preprocessing.text_chunker import split_documents


def generate_embeddings(folder_path):

    # Load embedding model
    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    # Get document chunks
    chunks = split_documents(folder_path)

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # Generate embeddings
    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    return chunks, embeddings


if __name__ == "__main__":

    folder = "data/raw"

    chunks, embeddings = generate_embeddings(folder)

    print(
        "\nTotal Chunks:",
        len(chunks)
    )

    print(
        "Embedding Shape:",
        embeddings.shape
    )