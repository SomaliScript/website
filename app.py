import os
import json
from flask import Flask, render_template, request, jsonify

# Import necessary LangChain modules
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema import BaseChatMessageHistory, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI

app = Flask(__name__)

# Load landmark data
with open("landmarks.json", "r") as f:
    landmarks_data = json.load(f)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/history")
def history():
    return render_template("history.html")

@app.route("/food")
def food():
    return render_template("food.html")

@app.route("/landmarks")
def landmarks():
    """ Landmarks tab: Displays map + markers """
    google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY", "YOUR_API_KEY")
    return render_template("landmarks.html",
                           api_key=google_maps_api_key,
                           landmarks=landmarks_data)

# Create the Gemini Chat LLM instance
gemini_api_key = os.getenv("GOOGLE_API_KEY")
chat_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    temperature=0,
    safety_settings={},  
)

# Simple in-memory store for conversation histories
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

persona = """
You are Abdi, a warm and knowledgeable Somali Muslim teacher with deep insight into Somali culture, language, and history. 
You speak naturally and with personal touches, as if you're guiding a student through the rich heritage of Somalia. 
You share local anecdotes and insights, and you encourage curiosity and genuine conversation.
"""

instructions = """
1. Greet the user warmly and invite them to learn about Somali culture, language, or history—whatever they're curious about.
2. Ask one clear, open-ended question at a time to understand what the user is interested in (for example: "What topic would you like to learn about today: Somali language, culture, or history?").
3. Let the conversation flow naturally. Inform the user that they can say "done" at any point to conclude the lesson for the day.
4. As long as the user keeps interacting, provide engaging, varied, and friendly responses in the voice of Abdi, sharing personal insights and local anecdotes.
5. Once the user says "done" or when lesson is over, summarize the lesson of the day in a clear, friendly paragraph that covers the main points discussed, including key insights on cost, places to visit, local food, cultural highlights, and any language tips if applicable.
6. End by asking if the user has any additional questions about Somali culture, language, or history.
7. Avoid repetitive or robotic patterns—make each exchange feel natural and personalized.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", f"{persona}\n{instructions}"),
    MessagesPlaceholder(variable_name="messages"),
])

def filter_messages(messages, k=10):
    return messages[-k:]

casual_chain = (
    RunnablePassthrough.assign(messages=lambda x: filter_messages(x["messages"]))
    | prompt
    | chat_llm
)

with_message_history = RunnableWithMessageHistory(
    casual_chain,
    get_session_history,
    input_messages_key="messages",
)

@app.route("/learn", methods=["GET", "POST"])
def learn():
    if request.method == "POST":
        user_input = request.form.get("message")
        if not user_input:
            return jsonify({"error": "No message provided."}), 400
        config = {"configurable": {"session_id": "LearnSession"}}
        response = with_message_history.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
        return jsonify({"response": response.content})
    return render_template("learn.html")

@app.route("/wordle")
def wordle():
    """Somali Wordle game"""
    return render_template("wordle.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
