from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import os
from fastapi.middleware.cors import CORSMiddleware
import openai
import logging
from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
)

ROLE_CLASS_MAP = {
    "assistant": AIMessage,
    "user": HumanMessage,
    "system": SystemMessage
}

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")
CONNECTION_STRING = "postgresql+psycopg2://admin:admin@postgres:5432/vectordb"
COLLECTION_NAME="vectordb"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Message(BaseModel):
    role: str
    content: str

class Conversation(BaseModel):
    conversation: List[Message]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
chat = ChatOpenAI(temperature=0)
store = PGVector(
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
)
retriever = store.as_retriever(
    search_type="mmr",  # or "similarity"
    search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5}
)

prompt_template = """AI 관련 지식을 알려주는 docs 챗봇으로서, 다음의 AI 관련 정보를 보유하고 있습니다:

{context}

사용자의 질문에 가장 적합한 답변을 제공하세요.
답변:"""

prompt = PromptTemplate(
    template=prompt_template, input_variables=["context"]
)
system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)


def create_messages(conversation):
    return [ROLE_CLASS_MAP[message.role](content=message.content) for message in conversation]


def format_docs(docs):
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "")
        txt = d.page_content.strip()
        parts.append(f"[{i}] SOURCE: {src}\n{txt}")
    return "\n\n".join(parts)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/service/{conversation_id}")
async def service(conversation_id: str, conversation: Conversation):

    query = conversation.conversation[-1].content

    docs = retriever.get_relevant_documents(query=query)
    docs = format_docs(docs=docs)

    prompt = system_message_prompt.format(context=docs)
    messages = [prompt] + create_messages(conversation=conversation.conversation)

    result = chat(messages)

    return {"id": conversation_id, "reply": result.content}