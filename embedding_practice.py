import requests
import math

# ----------------------------------
# 1. Configuration
# ----------------------------------

OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text"

# ----------------------------------
# 2. Sample sentences
# ----------------------------------

sentences = [
    "I love programming in Python.",
    "I enjoy writing code.",
    "I like playing cricket."
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
# 4. Calculate cosine similarity
# ----------------------------------

def cosine_similarity(vector_a, vector_b):

    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# ----------------------------------
# 5. Main program
# ----------------------------------

try:

    embeddings = generate_embeddings(sentences)

    print("\nEMBEDDINGS GENERATED SUCCESSFULLY!")

    print("\nCOSINE SIMILARITY RESULTS")

    # Compare every pair of sentences

    for i in range(len(sentences)):

        for j in range(i + 1, len(sentences)):

            similarity = cosine_similarity(
                embeddings[i],
                embeddings[j]
            )

            print("\nSentence 1:", sentences[i])
            print("Sentence 2:", sentences[j])

            print("Similarity:", round(similarity, 4))

except requests.exceptions.RequestException as error:

    print("Ollama connection error:", error)