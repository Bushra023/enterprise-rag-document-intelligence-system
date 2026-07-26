from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.ingestion.document_loader import load_pdf_documents


def split_documents(folder_path):
    # Load the PDF documents
    documents = load_pdf_documents(folder_path)

    # Create the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    all_chunks = []

    # Split each document into chunks
    for document in documents:

        chunks = text_splitter.split_text(document["text"])

        for chunk in chunks:
            all_chunks.append(
                {
                    "file_name": document["file_name"],
                    "text": chunk
                }
            )

    return all_chunks


if __name__ == "__main__":

    folder = "data/raw"

    chunks = split_documents(folder)

    print(f"\nTotal Chunks Created: {len(chunks)}\n")

    # Display the first five chunks
    for index, chunk in enumerate(chunks[:5], start=1):
        print(f"Chunk {index}")
        print(f"Document: {chunk['file_name']}")
        print(f"Characters: {len(chunk['text'])}")
        print("-" * 60)