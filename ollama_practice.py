
import requests
import json
from pathlib import Path

# Ollama API URL
url = "http://localhost:11434/api/chat"

# Model name
model = "llama3.2:1b"

# JSON conversation history file
history_file = Path("chat_history.json")

# System prompt: Defines AI personality and behaviour
system_prompt = {
    "role": "system",
    "content": """
    You are a friendly Generative AI tutor.

    Follow these instructions:
    1. Explain concepts in simple English.
    2. Give real-world examples.
    3. Explain technical terms clearly.
    4. Teach step by step.
    5. Keep answers beginner-friendly.
    6. Avoid unnecessarily complicated explanations.
    """
}

# Load previous conversation
if history_file.exists():

    try:
        with open(history_file, "r", encoding="utf-8") as file:
            messages = json.load(file)

        print("Previous conversation loaded!")

    except (json.JSONDecodeError, OSError):
        print("Could not load history. Starting fresh.")
        messages = []

else:
    messages = []

# Ensure system prompt is present
if not messages or messages[0].get("role") != "system":
    messages.insert(0, system_prompt)

print("\n================================")
print(" Welcome to My GenAI Chatbot")
print("================================")

print("AI Personality: Generative AI Tutor")
print("Type 'exit' to stop.")
print("Type 'clear' to clear conversation history.\n")


# Main chatbot loop
while True:

    # Get user question
    question = input("You: ")

    # Exit chatbot
    if question.lower().strip() == "exit":

        print("\nAI: Goodbye! Keep learning Generative AI.")
        break

    # Clear conversation history
    if question.lower().strip() == "clear":

        messages = [system_prompt]

        with open(history_file, "w", encoding="utf-8") as file:
            json.dump(messages, file, indent=4, ensure_ascii=False)

        print("\nAI: Conversation history cleared!\n")
        continue

    # Handle empty input
    if not question.strip():

        print("AI: Please enter a question.\n")
        continue

    # Add user question to conversation history
    messages.append({
        "role": "user",
        "content": question
    })

    # Prepare API request
    data = {
        "model": model,
        "messages": messages,
        "stream": False
    }

    try:

        # Send request to Ollama
        response = requests.post(
            url,
            json=data,
            timeout=120
        )

        # Check for HTTP errors
        response.raise_for_status()

        # Convert JSON response into Python dictionary
        result = response.json()

        # Extract AI response
        ai_reply = result["message"]["content"]

        print("\nAI:", ai_reply, "\n")

        # Save AI response in conversation history
        messages.append({
            "role": "assistant",
            "content": ai_reply
        })

        # Save conversation into JSON file
        with open(history_file, "w", encoding="utf-8") as file:

            json.dump(
                messages,
                file,
                indent=4,
                ensure_ascii=False
            )

    except requests.exceptions.RequestException as error:

        print("\nAPI Error:", error)

        # Remove unanswered user question
        messages.pop()

    except (KeyError, ValueError) as error:

        print("\nResponse processing error:", error)

        # Remove unanswered user question
        messages.pop()
