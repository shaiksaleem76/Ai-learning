import requests

try:
    response = requests.get(
        "https://jsonplaceholder.typicode.com/todos/1",
        timeout=10
    )

    response.raise_for_status()
    data = response.json()

    print("Task:", data["title"])
    print("Completed:", data["completed"])

except requests.exceptions.RequestException as error:
    print("API error:", error)