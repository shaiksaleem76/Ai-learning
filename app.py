
import streamlit as st
import requests
import json
from pathlib import Path

# Import RAG backend
from rag_chatbot import (
    retrieve_documents_with_distances,
    check_database
)

# ----------------------------------
# 1. Configuration
# ----------------------------------

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:1b"

HISTORY_FILE = Path("streamlit_chat_history.json")


# AI personality
SYSTEM_PROMPT = {
    "role": "system",
    "content": """
You are a friendly Generative AI tutor.

Follow these instructions:
1. Explain concepts in simple English.
2. Give real-world examples.
3. Explain technical terms clearly.
4. Teach step by step.
5. Keep answers beginner-friendly.
"""
}


# RAG instructions
RAG_SYSTEM_PROMPT = {
    "role": "system",
    "content": """
You are a context-based question answering assistant.

Follow these rules:
1. Answer using only the retrieved context.
2. If the context contains the answer, answer directly.
3. Do not include unrelated information.
4. Do not invent facts.
5. If the answer is missing, say:
   I don't have enough information in the provided documents.
6. Keep answers simple and clear.
"""
}


# ----------------------------------
# 2. Load conversation history
# ----------------------------------

def load_history():

    if HISTORY_FILE.exists():

        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as file:
                history = json.load(file)

            if not isinstance(history, list):
                history = []

        except (json.JSONDecodeError, OSError):
            history = []

    else:
        history = []

    # Keep only valid messages
    history = [
        message for message in history
        if isinstance(message, dict)
        and message.get("role") in ["user", "assistant"]
        and isinstance(message.get("content"), str)
    ]

    return [SYSTEM_PROMPT] + history


# ----------------------------------
# 3. Save conversation history
# ----------------------------------

def save_history(messages):

    history = [
        message for message in messages
        if message["role"] in ["user", "assistant"]
    ]

    with open(HISTORY_FILE, "w", encoding="utf-8") as file:

        json.dump(
            history,
            file,
            indent=4,
            ensure_ascii=False
        )


# ----------------------------------
# 4. Stream AI response with token usage
# ----------------------------------

def stream_ai_response(messages, usage_stats):

    data = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": 0
        }
    }

    with requests.post(
        OLLAMA_URL,
        json=data,
        stream=True,
        timeout=(10, 180)
    ) as response:

        response.raise_for_status()

        for line in response.iter_lines():

            if not line:
                continue

            chunk = json.loads(line)

            # Extract generated text
            content = chunk.get("message", {}).get("content", "")

            if content:
                yield content

            # Extract token statistics
            if chunk.get("done"):

                usage_stats["input_tokens"] = chunk.get(
                    "prompt_eval_count", 0
                )

                usage_stats["output_tokens"] = chunk.get(
                    "eval_count", 0
                )

                break


# ----------------------------------
# 5. Streamlit page configuration
# ----------------------------------

st.set_page_config(
    page_title="My RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 My RAG Chatbot")

st.caption(
    "Your AI tutor | RAG + ChromaDB + Ollama"
)


# ----------------------------------
# 6. Initialize session state
# ----------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = load_history()


# ----------------------------------
# 7. Sidebar settings
# ----------------------------------

with st.sidebar:

    st.header("Chat Settings")

    st.write("Chat Model:", MODEL)

    st.write("Embedding Model:", "nomic-embed-text")

    st.write("Vector Database:", "ChromaDB")

    st.write("Status: Local AI")

    if st.button(
        "Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = [SYSTEM_PROMPT]

        save_history(st.session_state.messages)

        st.success("Conversation cleared!")

        st.rerun()


# ----------------------------------
# 8. Display previous messages
# ----------------------------------

for message in st.session_state.messages:

    if message["role"] == "system":
        continue

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ----------------------------------
# 9. User input
# ----------------------------------

question = st.chat_input(
    "Ask a question about your documents..."
)


if question:

    # Display user question
    with st.chat_message("user"):

        st.markdown(question)

    # Add user question to history
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    usage_stats = {}

    try:

        # ----------------------------------
        # Step 1: Check ChromaDB
        # ----------------------------------

        if not check_database():

            raise ValueError(
                "No documents found in ChromaDB. "
                "Run vector_database.py first."
            )


        # ----------------------------------
        # Step 2: Retrieve relevant documents
        # ----------------------------------

        with st.spinner("Searching relevant documents..."):

            context, distances = retrieve_documents_with_distances(
                question
            )


        # ----------------------------------
        # Step 3: Check filtered results
        # ----------------------------------

        if not context:

            raise ValueError(
                "No sufficiently relevant documents found. "
                "Please try another question."
            )


        # ----------------------------------
        # Step 4: Display retrieved context
        # ----------------------------------

        with st.expander("📚 View Retrieved Context"):

            for index, (document, distance) in enumerate(
                zip(context, distances),
                start=1
            ):

                st.markdown(f"**Document {index}**")

                st.write(document)

                st.caption(
                    f"Distance: {distance:.4f}"
                )

                st.divider()


        # ----------------------------------
        # Step 5: Prepare conversation for RAG
        # ----------------------------------

        # Keep previous conversation messages
        previous_messages = st.session_state.messages[1:-1]

        # Prepare retrieved context
        context_text = "\n".join(
            f"- {document}" for document in context
        )

        rag_question = f"""
RETRIEVED CONTEXT:

{context_text}

USER QUESTION:

{question}

Answer the question using only the retrieved context.
"""


        # Prepare messages for Ollama
        api_messages = [
            SYSTEM_PROMPT,
            RAG_SYSTEM_PROMPT
        ]

        api_messages.extend(previous_messages)

        api_messages.append({
            "role": "user",
            "content": rag_question
        })


        # ----------------------------------
        # Step 6: Generate streaming answer
        # ----------------------------------

        with st.chat_message("assistant"):

            ai_reply = st.write_stream(
                stream_ai_response(
                    api_messages,
                    usage_stats
                )
            )


        # ----------------------------------
        # Step 7: Save AI response
        # ----------------------------------

        st.session_state.messages.append({
            "role": "assistant",
            "content": ai_reply
        })

        save_history(st.session_state.messages)


        # ----------------------------------
        # Step 8: Display token usage
        # ----------------------------------

        input_tokens = usage_stats.get(
            "input_tokens", 0
        )

        output_tokens = usage_stats.get(
            "output_tokens", 0
        )

        total_tokens = input_tokens + output_tokens

        st.divider()

        st.subheader("📊 Token Usage")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Input Tokens",
                input_tokens
            )

        with col2:

            st.metric(
                "Output Tokens",
                output_tokens
            )

        with col3:

            st.metric(
                "Total Tokens",
                total_tokens
            )


    # ----------------------------------
    # 9. Error handling
    # ----------------------------------

    except requests.exceptions.RequestException as error:

        st.session_state.messages.pop()

        st.error(
            f"Ollama connection error: {error}"
        )

    except ValueError as error:

        st.session_state.messages.pop()

        st.error(
            f"Application error: {error}"
        )

    except Exception as error:

        st.session_state.messages.pop()

        st.error(
            f"Unexpected error: {error}"
        )
