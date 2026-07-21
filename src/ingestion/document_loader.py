import os
from pypdf import PdfReader


def load_pdf_documents(folder_path):
    documents = []

    for file_name in os.listdir(folder_path):

        if file_name.endswith(".pdf"):

            file_path = os.path.join(folder_path, file_name)

            print(f"Processing: {file_name}")

            pdf_reader = PdfReader(file_path)

            text = ""

            for page in pdf_reader.pages:
                page_text = page.extract_text()

                if page_text:
                    text += page_text

            documents.append(
                {
                    "file_name": file_name,
                    "text": text
                }
            )

    return documents


if __name__ == "__main__":

    pdf_folder = "data/raw"

    documents = load_pdf_documents(pdf_folder)

    print("\nTotal Documents Loaded:", len(documents))

    for doc in documents:
        print(
            doc["file_name"],
            "Characters:",
            len(doc["text"])
        )