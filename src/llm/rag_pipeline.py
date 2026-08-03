from ollama import Client

from src.retrieval.semantic_search import search_documents
client = Client(
    host="http://host.docker.internal:11434"
)

def build_context(results):

    context = ""

    for result in results:

        context += f"""
Document: {result['file_name']}

{result['text']}

-----------------------------
"""

    return context



def ask_rag(question):

    # Step 1: Retrieve relevant chunks from FAISS
    results = search_documents(question)

    # Step 2: Build context from retrieved documents
    context = build_context(results)


    # Step 3: Create prompt for Llama
    prompt = f"""
You are an enterprise document assistant.

Answer the question ONLY using the information provided in the context below.

If the answer is not available in the context, say:
"I could not find this information in the provided documents."

Context:

{context}


Question:

{question}


Answer:
"""


    # Step 4: Send prompt to Llama 3.1 through Ollama
    response = client.chat(
        model="llama3.1",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )


    # Step 5: Return only the answer
    return response["message"]["content"]



if __name__ == "__main__":

    question = input("Enter your question: ")

    answer = ask_rag(question)

    print("\nAnswer:\n")

    print(answer)
