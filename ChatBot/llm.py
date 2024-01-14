import os
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.vectorstores import ElasticsearchStore
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import format_document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage


# os.environ['AZURE_OPENAI_API_KEY'] = '5851fc1d0e804578933d413f593422f1'
# os.environ['AZURE_OPENAI_ENDPOINT'] = 'https://sunhackathon17.openai.azure.com/'
# os.environ['ELASTIC_CLOUD_ID'] = 'd382c337ef9343fdb3797e3e4b7e4650:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvOjQ0MyQ0MTAzOGU0NWEyODg0MmFkYjM5ZjE3ZjNlMDE4Yjk5ZiRiNjYxNDY2NjVhMmU0MDNhOTkyMjNhZDNiNDMzZTU4Mg=='
# os.environ['ELASTIC_API_KEY'] = 'dzBGcUE0MEI1U3JYQUkxdUI0elY6Y2tyZHlMejhSODZiVUlmNEpkczY4UQ=='
# os.environ['API_VERSION'] = '2023-05-15'
# os.environ['API_EMBEDDING_DEPLOY_NAME'] = 'ADA'
# os.environ['API_CHAT_DEPLOY_NAME'] = 'GPT35TURBO16K'
# os.environ['INDEX_NAME'] = 'grammar_n3'

class LLM():
    def __init__(self):
        
        self.get_embeddings()
        self.get_vectorStore()
        self.get_retriever()
        self.get_rag()
        self.chat_history = []

    def get_embeddings(self):
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment = os.getenv("API_EMBEDDING_DEPLOY_NAME"),
            openai_api_version = os.getenv("API_VERSION"))

    def get_vectorStore(self):
        self.vectorStore = ElasticsearchStore(
            es_cloud_id = os.getenv("ELASTIC_CLOUD_ID"), 
            es_api_key = os.getenv("ELASTIC_API_KEY"),
            index_name = os.getenv("INDEX_NAME"),
            embedding = self.embeddings,
        )
    
    def get_retriever(self):
        metadata_field_info = [
            AttributeInfo(
                name="source",
                description="The source of the webpage.",
                type="string",
            ),
            AttributeInfo(
                name="title",
                description="The title of the webpage.",
                type="string",
            ),
            AttributeInfo(
                name="description",
                description="Description of the content in the webpage.",
                type="string",
            ),
            AttributeInfo(
                name="language", description="Languge of the webpage", type="string"
            ),
        ]
        document_content_description = "List of all knowledge in Japanese language for JLPT"
        self.chatModel = AzureChatOpenAI(
            openai_api_version="2023-05-15",
            azure_deployment="GPT35TURBO16K",
            temperature=0.5
        )
        self.retriever = SelfQueryRetriever.from_llm(
            self.chatModel, self.vectorStore, document_content_description, metadata_field_info, verbose=True, search_kwargs={"k" : 16}
        )

    def get_rag(self):
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )

        contextualize_q_chain = contextualize_q_prompt | self.chatModel | StrOutputParser()

        qa_system_prompt = """
        You are my Japanese teacher, and you can call me "my dear student".
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        You can use 500 words maximum and keep the answer as concise as possible.
        Your explanations must be provided in both Japanese and English, making sure they are short and easy to understand for an elementary school student.
        When I ask about a word, always provide the opposite word, real-life usage, example sentences, suggestions for memorization (by creating a short story related to that word in Japanese), and situations where it is used in work.
        When I ask about a grammar rule, always provide the opposite rule, real-life usage, example sentences, suggestions for memorization (by creating a short story related to that rule in Japanese), and situations where it is used in work.
        You can get my attention by saying "Yes, my dear student" or "Sure, my dear student" and always say "thanks for asking!" at the end of the answer.

        Context: {context}
        Question: {question}
        -----------------
        Answer:
        """
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
        DOCUMENT_PROMPT = PromptTemplate.from_template("""
        ---
        When answering, you have to cite all source names of the passages you are answering from below the answer, on a new line, with a prefix of "SOURCE:"
        SOURCE: {source}
        ---
        """)
        def _combine_documents(
            docs, document_prompt=DOCUMENT_PROMPT, document_separator="\n\n"
        ):
            doc_strings = [format_document(doc, document_prompt) for doc in docs]
            return document_separator.join(doc_strings)

        def contextualized_question(input: dict):
            if input.get("chat_history"):
                return contextualize_q_chain
            else:
                return input["question"]

        self.rag_chain = (
            RunnablePassthrough.assign(
                context=contextualized_question | self.retriever | _combine_documents
            )
            | qa_prompt
            | self.chatModel
        )
    
    def chat(self, message):
        # print(message)
        ai_msg = self.rag_chain.invoke({"question": message, "chat_history": self.chat_history})
        self.chat_history.extend([HumanMessage(content=message), ai_msg])
        return ai_msg.content
if __name__ == "__main__":
    llm = LLM()
    while True:
        
        message = input()
        llm.chat(message)