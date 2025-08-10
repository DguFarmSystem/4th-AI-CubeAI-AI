from typing import List
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import os
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import Literal
from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain_core.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
)

ROLE_CLASS_MAP = {
    "assistant": AIMessage,
    "user": HumanMessage,
    "system": SystemMessage
}

load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

CONNECTION_STRING = os.getenv("CONNECTION_STRING", "postgresql+psycopg2://admin:admin@postgres:5432/vectordb")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "vectordb")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(..., min_length=1, max_length=10000)

class Conversation(BaseModel):
    conversation: List[Message] = Field(..., min_items=1, max_items=50)

try:
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL, api_key=OPENAI_API_KEY)
    chat = ChatOpenAI(temperature=TEMPERATURE, max_tokens=MAX_TOKENS, api_key=OPENAI_API_KEY)
    store = PGVector(
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
    )
    retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5}
    )
    logger.info(f"Successfully initialized vector store with collection: {COLLECTION_NAME}")
except Exception as e:
    logger.error(f"Failed to initialize vector store: {e}")
    raise

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

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "openai-rag"}

@app.post("/service/{conversation_id}")
async def service(conversation_id: str, conversation: Conversation):
    try:
        logger.info(f"Processing conversation {conversation_id}")
        
        query = conversation.conversation[-1].content
        logger.debug(f"Query: {query[:100]}...")
        
        docs = retriever.get_relevant_documents(query=query)
        if not docs:
            logger.warning(f"No relevant documents found for query: {query[:50]}...")
            
        formatted_docs = format_docs(docs=docs)
        logger.debug(f"Retrieved {len(docs)} documents")
        
        prompt = system_message_prompt.format(context=formatted_docs)
        messages = [prompt] + create_messages(conversation=conversation.conversation)
        
        result = chat(messages)
        logger.info(f"Successfully generated response for conversation {conversation_id}")
        
        return {"id": conversation_id, "reply": result.content}
        
    except Exception as e:
        logger.error(f"Error processing conversation {conversation_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process conversation: {str(e)}"
        )