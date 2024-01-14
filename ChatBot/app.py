from flask import Flask, render_template, request, jsonify
import warnings
warnings.filterwarnings("ignore")
import os
#dùng langchain
from llm import LLM
from langchain_openai import AzureChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
#khởi tạo llm instance
llm = LLM()
#code api
app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)


def get_Chat_response(text):

    while True:
        # user_input = text
        # user_message = HumanMessage(content=user_input)
        # messages.append(user_message)

        response = llm.chat(text)

        #messages.append(AIMessage(content=response))
        return response
    

if __name__ == '__main__':
    app.run()
