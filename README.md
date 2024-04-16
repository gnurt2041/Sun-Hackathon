# Sun* Hackathon - JPT - Your AI Japanese teacher 
<p align="center">
<br><br><br>
<a https://github.com/Haste171/langchain-chatbot/stargazers"><img src="https://i.imgur.com/9Caf7yp.png" width="760px" length="400"></a>
<br><br><br>
</p>

<p align="center">
<b>Efficiently use Langchain for learning Japanese</b>

<!-- *The LangChain Chatbot is an AI chat interface for the open-source library LangChain. It provides conversational answers to questions about vector ingested documents.* -->
<!-- *Existing repo development is at a freeze while we develop a langchain chat bot website :)* -->


# 🚀 Installation

### Setup
```
git clone https://github.com/gnurt2041/Sun-Hackathon.git
```

Install the requirements packages
```python
cd ChatBot
pip install -r requirements.txt
```

Reference [config.sh](https://github.com/gnurt2041/Sun-Hackathon/blob/main/ChatBot/config.sh) to create `config.sh` file
```python
export AZURE_OPENAI_API_KEY = "YOUR_AZURE_OPENAI_API_KEY"
export AZURE_OPENAI_ENDPOINT = "YOUR_AZURE_OPENAI_ENDPOINT"
export API_VERSION = "YOUR_API_VERSION"
export API_EMBEDDING_DEPLOY_NAME = "YOUR_API_EMBEDDING_DEPLOY_NAME"
export API_CHAT_DEPLOY_NAME = "YOUR_API_CHAT_DEPLOY_NAME"
export ELASTIC_CLOUD_ID = "YOUR_ELASTIC_CLOUD_ID"
export ELASTIC_API_KEY = "YOUR_ELASTIC_API_KEY"
export INDEX_NAME = "YOUR_INDEX_NAME"
```

Run app in local host

```python
python app.py
```

# 📝 Credits

The LangChain  Chatbot was developed by [Gnurt2041](https://github.com/gnurt2041), [Minhquan129](https://github.com/Minhquan129) and [Omnihs1](https://github.com/Omnihs1) with much inspiration from [ELSA Speak](https://vn.elsaspeak.com/en/homepage/) with the [LangChain Q&A with RAG ](https://python.langchain.com/docs/use_cases/question_answering/) for PDF docs and content from web page.
