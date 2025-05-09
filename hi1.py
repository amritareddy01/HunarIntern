import datetime

def get_response(user_input):
    user_input = user_input.lower()

    if any(greeting in user_input for greeting in ["hello", "hi", "hey"]):
        return "Hi there! How can I assist you today?"
    elif "how are you" in user_input:
        return "I'm doing great, thanks for asking! How about you?"
    elif "help" in user_input:
        return "Sure! You can ask me about the time, weather, or just talk to me."
    elif "weather" in user_input:
        return "I'm not connected to live weather data, but I hope it's sunny where you are! ☀️"
    elif "time" in user_input:
        now = datetime.datetime.now().strftime("%I:%M %p")
        return f"The current time is {now} ⏰"
    elif "date" in user_input:
        today = datetime.date.today().strftime("%B %d, %Y")
        return f"Today's date is {today} 📅"
    elif "bye" in user_input or "exit" in user_input or "quit" in user_input:
        return "Goodbye! It was nice chatting with you. Have a great day!:)"
    else:
        return "Hmm, I didn't quite catch that. Could you rephrase or ask something else?"

def chatbot():
    print("Chatty: Hi! I'm your rule-based chatbot. Type 'bye' to end the chat.\n")

    while True:
        user_input = input("You: ").strip()
        response = get_response(user_input)
        print(f"Chatty: {response}")

        if "bye" in user_input.lower() or "exit" in user_input.lower() or "quit" in user_input.lower():
            break

# Start the chatbot
chatbot()
