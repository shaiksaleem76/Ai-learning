
import requests
import chromadb

# ----------------------------------
# 1. Configuration
# ----------------------------------

OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2:1b"

DB_PATH = "./chroma_db"

# Maximum number of documents to retrieve
TOP_K = 3

# Maximum allowed distance for a document
DISTANCE_THRESHOLD = 0.75


# ----------------------------------
# 2. Connect to ChromaDB
# ----------------------------------

client = chromadb.PersistentClient(
    path=DB_PATH
)

collection = client.get_or_create_collection(
    name="ai_learning_documents"
)


# ----------------------------------
# 3. Check database
# ----------------------------------

def check_database():

    return collection.count() > 0


# ----------------------------------
# 4. Generate embeddings
# ----------------------------------

def generate_embedding(text):

    response = requests.post(
        OLLAMA_EMBED_URL,
        json={
            "model": EMBED_MODEL,
            "input": [text]
        },
        timeout=120
    )

    response.raise_for_status()

    embedding = response.json()["embeddings"][0]

    return embedding


# ----------------------------------
# 5. Retrieve documents with distances
# ----------------------------------

def retrieve_documents_with_distances(question):

    # Check whether database contains documents
    if not check_database():

        return [], []

    # Generate question embedding
    query_embedding = generate_embedding(question)

    # Get the number of documents to retrieve
    number_of_results = min(
        TOP_K,
        collection.count()
    )

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=number_of_results
    )

    # Extract documents and distances
    documents = results["documents"][0] or []

    distances = results["distances"][0] or []

    # ----------------------------------
    # 6. Similarity threshold filtering
    # ----------------------------------

    filtered_documents = []
    filtered_distances = []

    for document, distance in zip(
        documents,
        distances
    ):

        # Keep only relevant documents
        if distance <= DISTANCE_THRESHOLD:

            filtered_documents.append(document)

            filtered_distances.append(distance)

    return filtered_documents, filtered_distances


# ----------------------------------
# 7. Generate answer using Llama
# ----------------------------------

def generate_answer(question, context):

    # Check whether relevant context exists
    if not context:

        return (
            "I don't have enough information "
            "in the provided documents."
        )

    # Prepare context
    context_text = "\n".join(
        f"- {document}" for document in context
    )

    system_prompt = """
You are a helpful question-answering assistant.

Answer the user's question using only the supplied context.

Rules:
1. Read the context carefully.
2. If the answer is present, answer directly.
3. Do not include unrelated information.
4. Do not invent facts.
5. If the answer is missing, say:
   I don't have enough information in the provided documents.
6. Keep the answer simple and clear.
"""

    user_prompt = f"""
CONTEXT:
{context_text}

QUESTION:
{question}

Give only the answer:
"""

    # Send request to Ollama
    response = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": CHAT_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0
            }
        },
        timeout=180
    )

    response.raise_for_status()

    answer = response.json()["message"]["content"].strip()

    return answer


# ----------------------------------
# 8. Main RAG function
# ----------------------------------

def answer_question(question):

    # Step 1: Retrieve filtered documents
    context, distances = retrieve_documents_with_distances(
        question
    )

    # Step 2: Check retrieved context
    if not context:

        answer = (
            "I don't have enough information "
            "in the provided documents."
        )

        return answer, context, distances

    # Step 3: Generate answer
    answer = generate_answer(
        question,
        context
    )

    # Step 4: Return results
    return answer, context, distances
