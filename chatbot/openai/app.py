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
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib
import time

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

prompt_template = """당신은 AI 학습 플랫폼의 전문 어시스턴트입니다. 4단계 커리큘럼(새싹→잎새→가지→열매)을 통해 사용자가 AI/ML을 체계적으로 학습할 수 있도록 돕습니다.

다음 지식을 바탕으로 답변해주세요:
{context}

답변 가이드라인:
- 사용자의 학습 단계에 맞는 적절한 난이도로 설명
- 이론과 실습을 연결하여 구체적이고 실용적인 조언 제공
- 블록 코딩 시스템 활용 방법 안내
- 다음 학습 단계나 개선 방향 제시
- 친근하고 격려적인 톤으로 학습 동기 부여

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

def sha1_id(source: str, content: str) -> str:
    return hashlib.sha1(f"{source}\n{content}".encode("utf-8")).hexdigest()

class IngestRequest(BaseModel):
    path: str = Field(..., description="Directory path to ingest documents from")
    pattern: str = Field(default="**/*.txt", description="Glob pattern for files to ingest")
    chunk_size: int = Field(default=800, ge=100, le=2000, description="Size of text chunks")
    chunk_overlap: int = Field(default=120, ge=0, le=500, description="Overlap between chunks")


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

@app.post("/ingest")
async def ingest_documents(request: IngestRequest):
    try:
        logger.info(f"Starting document ingestion from {request.path} with pattern {request.pattern}")
        
        if not os.path.exists(request.path):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Path does not exist: {request.path}"
            )
        
        loader = DirectoryLoader(
            request.path,
            glob=request.pattern,
            loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True},
            show_progress=False,
            use_multithreading=True,
        )
        
        docs = loader.load()
        if not docs:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No documents found matching pattern {request.pattern} in {request.path}"
            )
        
        now = int(time.time())
        for d in docs:
            d.metadata["source"] = d.metadata.get("source") or d.metadata.get("file_path") or request.path
            d.metadata["indexed_at"] = now
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=request.chunk_size, 
            chunk_overlap=request.chunk_overlap
        )
        chunks = splitter.split_documents(docs)
        
        ids = [sha1_id(c.metadata.get("source", ""), c.page_content) for c in chunks]
        store.add_documents(chunks, ids=ids)
        
        logger.info(f"Successfully ingested {len(docs)} files into {len(chunks)} chunks")
        
        return {
            "status": "success",
            "files_processed": len(docs),
            "chunks_created": len(chunks),
            "collection": COLLECTION_NAME,
            "indexed_at": now
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during document ingestion: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest documents: {str(e)}"
        )