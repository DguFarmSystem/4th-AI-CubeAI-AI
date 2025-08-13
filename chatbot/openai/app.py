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

def is_vague_query(query: str) -> bool:
    """사용자의 질문이 모호한지 판단"""
    vague_indicators = [
        # 일반적인 모호한 표현들
        "도움", "알려", "설명", "뭔가", "어떻게", "무엇", "어떤", "좀", "혹시",
        # 불완전한 질문들
        "에 대해", "관련", "대한", "에서", "쪽으로", "관해서",
        # 애매한 단어들
        "것", "거", "그", "이", "저", "그런", "이런", "저런"
    ]
    
    # 질문이 너무 짧거나 (5글자 이하)
    if len(query.strip()) <= 5:
        return True
    
    # 구체적인 용어가 없고 모호한 표현만 있는 경우
    vague_count = sum(1 for indicator in vague_indicators if indicator in query)
    words = query.split()
    
    # 전체 단어 중 모호한 표현의 비율이 높으면 모호한 질문으로 판단
    if len(words) > 0 and vague_count / len(words) > 0.3:
        return True
    
    # 질문표가 없고 서술형인 경우도 모호할 수 있음
    if "?" not in query and len(words) < 3:
        return True
        
    return False

def generate_clarifying_questions(query: str, conversation_history: List[Message]) -> str:
    """모호한 질문에 대한 명확화 질문들을 생성"""
    
    # 대화 히스토리에서 맥락 파악
    context_prompt = ""
    if len(conversation_history) > 1:
        recent_context = conversation_history[-3:]  # 최근 3개 메시지만 사용
        context_prompt = f"최근 대화 맥락:\n"
        for msg in recent_context[:-1]:  # 마지막 질문 제외
            context_prompt += f"- {msg.role}: {msg.content}\n"
    
    clarifying_prompt = f"""사용자의 질문이 모호합니다. AI 학습 플랫폼 맥락에서 더 구체적인 도움을 제공하기 위해 명확화 질문을 생성해주세요.

{context_prompt}

사용자 질문: "{query}"

다음 영역 중 어떤 것에 관심이 있는지 2-3개의 선택지를 제공해주세요:

1. 데이터 전처리 관련 (데이터 선택, 정규화, 증강 등)
2. 모델 설계 관련 (CNN 구조, 활성화 함수, 레이어 설정 등)  
3. 학습 및 평가 관련 (훈련 과정, 성능 측정, 하이퍼파라미터 등)
4. 블록 코딩 시스템 사용법
5. 특정 오류나 문제 해결

친근하고 도움이 되는 톤으로 답변해주세요."""

    try:
        clarifying_messages = [HumanMessage(content=clarifying_prompt)]
        result = chat(clarifying_messages)
        return result.content
    except Exception as e:
        logger.error(f"Error generating clarifying questions: {e}")
        return """질문을 더 구체적으로 해주시면 더 정확한 도움을 드릴 수 있어요! 

어떤 영역에 대해 궁금하신가요?
1. 데이터 전처리 (데이터 준비, 정리, 변환)
2. 모델 설계 (신경망 구조, 레이어 설정)
3. 모델 훈련 및 평가 (학습 과정, 성능 측정)
4. 블록 코딩 시스템 사용법
5. 특정 문제나 오류 해결

구체적인 상황이나 목표를 알려주시면 더 맞춤형 조언을 드릴게요!"""

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
        
        # 모호한 질문 감지 및 명확화 질문 생성
        if is_vague_query(query):
            logger.info(f"Detected vague query, generating clarifying questions")
            clarifying_response = generate_clarifying_questions(query, conversation.conversation)
            return {"id": conversation_id, "reply": clarifying_response}
        
        # 구체적인 질문의 경우 기존 RAG 로직 수행
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