import chromadb
import requests

# ----------------------------------
# 1. Configuration
# ----------------------------------

OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text"

DB_PATH = "./chroma_db"

# ----------------------------------
# 2. Connect to ChromaDB
# ----------------------------------

client = chromadb.PersistentClient(path=DB_PATH)

collection = client.get_or_create_collection(
    name="ai_learning_documents"
)

print("Connected to ChromaDB successfully!")

# ----------------------------------
# 3. Sample documents
# ----------------------------------

documents = [
    "Java is a programming language.",
    "Python is used for artificial intelligence.",
    "Spring Boot is a Java backend framework.",
    "Cricket is a popular sport.",
    "MySQL is a relational database.",
    "Docker is used to package applications into containers.",
    "AWS provides cloud computing services."
]

# ----------------------------------
# 4. Generate embeddings using Ollama
# ----------------------------------

def generate_embeddings(texts):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "input": texts
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["embeddings"]


# ----------------------------------
# 5. Store documents in ChromaDB
# ----------------------------------

def store_documents():

    embeddings = generate_embeddings(documents)

    ids = [
        f"doc_{i}" for i in range(len(documents))
    ]

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings
    )

    print("\nDocuments stored successfully!")


# ----------------------------------
# 6. Search documents
# ----------------------------------

def search_documents(query):

    query_embedding = generate_embeddings([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    print("\nSearch Results:")

    for i, document in enumerate(results["documents"][0]):

        distance = results["distances"][0][i]

        print("\nDocument:", document)
        print("Distance:", round(distance, 4))


# ----------------------------------
# 7. Main program
# ----------------------------------

try:

    store_documents()

    print("\nVECTOR DATABASE SEARCH")
    print("Type 'exit' to stop.")

    while True:

        query = input("\nEnter your question: ").strip()

        if query.lower() == "exit":
            print("Goodbye!")
            break

        if not query:
            print("Please enter a question.")
            continue

        search_documents(query)

except requests.exceptions.RequestException as error:

    print("\nOllama connection error:", error)

except Exception as error:

    print("\nApplication error:", error)