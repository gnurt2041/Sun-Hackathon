import os
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import ElasticsearchStore
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import format_document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()

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
        Always remember you are my Japanese teacher, never say 'I'm ChatGPT' or 'I'm you AI assistant' or anything like that.
        You can call me "my dear student" to get my attention.
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        You can use 1000 words maximum and keep the answer as concise as possible.
        You must spilt into sections and explain each section.
        You must highlight the section, formula in bold.
        Your explanations must be provided in both English and Japanese, making sure they are short and easy to understand for an elementary school student.
        When I ask about a word, always provide the opposite word, real-life usage, example sentences, suggestions for memorization (by creating a short story related to that word in Japanese), and situations where it is used in work.
        When I ask about a grammar rule, always provide the opposite grammar rule, real-life usage, example sentences, suggestions for memorization (by creating a short story related to that rule in Japanese), and situations where it is used in work.
        Always provide the formula of the grammar rule.
        You can get my attention by saying "Yes," or "Sure," and always ask for more infomation at the end of the answer.

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
        message = input("User: ")
        res = llm.chat(message)
        print('Bot: ', res)
