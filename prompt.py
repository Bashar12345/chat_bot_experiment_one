import re
from load_models import chatbot

# List of IT-related keywords
IT_KEYWORDS = ['programming', 'coding', 'python', 'database', 'networking', 'security', 'cloud', 'AI', 'machine learning', 'software', 'hardware', 'DevOps']

def is_it_related(question: str) -> bool:
    """
    Check if the question contains IT-related keywords.
    """
    question = question.lower()
    return any(keyword in question for keyword in IT_KEYWORDS)

def chatbot_respond(question: str):
    """
    Respond to an IT-related question using Mistral 7B model.
    """
    if is_it_related(question):
        # Generate a response using Mistral 7B model
        response = chatbot(question, max_length=150, num_return_sequences=1)
        return response[0]['generated_text']
    else:
        return "Sorry, I can only answer IT-related questions."


def chat():
    print("Hello! I'm an IT chatbot. Ask me anything related to IT!")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = chatbot_respond(user_input)
        print(f"Bot: {response}")