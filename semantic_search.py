import requests
import math

# ----------------------------------
# 1. Configuration
# ----------------------------------

OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text"

# ----------------------------------
# 2. Our document collection
# ----------------------------------

documents = [
    "Java is a programming language.",
    "Python is used for artificial intelligence.",
    "Spring Boot is a Java backend framework.",
    "Cricket is a popular sport.",
    "MySQL is a relational database."
]

# ----------------------------------
# 3. Generate embeddings
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
# 4. Cosine similarity
# ----------------------------------

def cosine_similarity(vector_a, vector_b):

    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# ----------------------------------
# 5. Semantic search
# ----------------------------------

def semantic_search(query, documents, document_embeddings):

    # Convert user query into an embedding
    query_embedding = generate_embeddings([query])[0]

    results = []

    # Compare query with every document
    for index, document in enumerate(documents):

        score = cosine_similarity(
            query_embedding,
            document_embeddings[index]
        )

        results.append({
            "document": document,
            "score": score
        })

    # Sort by highest similarity
    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results


# ----------------------------------
# 6. Main program
# ----------------------------------

try:

    print("Generating document embeddings...")

    # Generate document embeddings only once
    document_embeddings = generate_embeddings(documents)

    print("Documents indexed successfully!")

    print("\nSEMANTIC SEARCH ENGINE")
    print("Type 'exit' to stop.")

    while True:

        query = input("\nEnter your question: ").strip()

        if query.lower() == "exit":
            print("Goodbye!")
            break

        if not query:
            print("Please enter a question.")
            continue

        results = semantic_search(
            query,
            documents,
            document_embeddings
        )

        print("\nMost relevant results:")

        for result in results[:3]:

            print("\nDocument:", result["document"])
            print("Similarity:", round(result["score"], 4))

except requests.exceptions.RequestException as error:

    print("Ollama connection error:", error)