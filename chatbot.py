import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.join(os.getcwd(), "intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Chatbot function with fallback response
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    try:
        tag = clf.predict(input_text)[0]
        for intent in intents:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
    except:
        return "I'm sorry, I don't understand. Can you rephrase?"

counter = 0

def main():
    global counter
    st.title("Intents of Chatbot using NLP")

    # Sidebar menu
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Menu
    if choice == "Home":
        st.write("Welcome! Type a message and press Enter to start.")

        # Check if chat_log.csv exists
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:
            user_input_str = str(user_input)
            response = chatbot(user_input)

            st.write(f"**Chatbot:** {response}")

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Save conversation history
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    # Conversation History Menu
    elif choice == "Conversation History":
        st.header("Conversation History")
        with st.expander("Click to see Conversation History"):
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip header
                for row in csv_reader:
                    st.text(f"User: {row[0]}")
                    st.text(f"Chatbot: {row[1]}")
                    st.text(f"Timestamp: {row[2]}")
                    st.markdown("---")

    elif choice == "About":
        st.write("This chatbot understands user intent using NLP and Logistic Regression.")
        st.subheader("Project Overview:")
        st.write("""
        1. NLP techniques and Logistic Regression are used to classify user intent.
        2. Streamlit provides a web-based chatbot interface.
        """)
        st.subheader("Dataset:")
        st.write("""
        - **Intents**: User intent categories (e.g., "greeting", "budget").
        - **Entities**: Extracted keywords (e.g., "Hi", "How do I create a budget?").
        - **Text**: User messages.
        """)
        st.subheader("Conclusion:")
        st.write("This chatbot can be improved with deep learning and a larger dataset.")

if __name__ == '__main__':
    main()