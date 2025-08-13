from typing import List
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
import redis
import requests
import json
import logging
import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
OPENAI_SERVICE_URL = os.getenv("OPENAI_SERVICE_URL", "http://openai:80")
DEFAULT_SYSTEM_MESSAGE = os.getenv("DEFAULT_SYSTEM_MESSAGE", "You are a helpful AI assistant.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    r.ping()  # Test connection
    logger.info(f"Successfully connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
except Exception as e:
    logger.error(f"Failed to connect to Redis: {e}")
    raise

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(..., min_length=1, max_length=10000)

class Conversation(BaseModel):
    conversation: List[Message] = Field(..., min_items=1, max_items=50)


@app.get("/health")
async def health_check():
    try:
        r.ping()
        return {"status": "healthy", "service": "conversation", "redis": "connected"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Redis connection failed: {str(e)}"
        )

@app.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    try:
        logger.info(f"Retrieving conversation {conversation_id}")
        existing_conversation_json = r.get(conversation_id)
        if existing_conversation_json:
            existing_conversation = json.loads(existing_conversation_json)
            logger.debug(f"Found conversation {conversation_id} with {len(existing_conversation.get('conversation', []))} messages")
            return existing_conversation
        else:
            logger.warning(f"Conversation {conversation_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation {conversation_id} not found"
            )
    except redis.RedisError as e:
        logger.error(f"Redis error retrieving conversation {conversation_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database temporarily unavailable"
        )
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in conversation {conversation_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Conversation data corrupted"
        )


@app.post("/conversation/{conversation_id}")
async def send_conversation(conversation_id: str, conversation: Conversation):
    try:
        logger.info(f"Processing conversation {conversation_id} with {len(conversation.conversation)} messages")
        
        # Retrieve existing conversation
        try:
            existing_conversation_json = r.get(conversation_id)
            if existing_conversation_json:
                existing_conversation = json.loads(existing_conversation_json)
                logger.debug(f"Found existing conversation {conversation_id}")
            else:
                existing_conversation = {"conversation": [{"role": "system", "content": DEFAULT_SYSTEM_MESSAGE}]}
                logger.debug(f"Created new conversation {conversation_id}")
        except (redis.RedisError, json.JSONDecodeError) as e:
            logger.error(f"Error retrieving conversation {conversation_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to retrieve conversation history"
            )

        # Add new user message
        new_message = conversation.dict()["conversation"][-1]
        existing_conversation["conversation"].append(new_message)
        logger.debug(f"Added user message to conversation {conversation_id}")

        # Forward to OpenAI service
        try:
            response = requests.post(
                f"{OPENAI_SERVICE_URL}/service/{conversation_id}", 
                json=existing_conversation,
                timeout=30
            )
            response.raise_for_status()
            assistant_message = response.json()["reply"]
            logger.debug(f"Received response from OpenAI service for conversation {conversation_id}")
        except requests.RequestException as e:
            logger.error(f"Error calling OpenAI service for conversation {conversation_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AI service temporarily unavailable"
            )
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Invalid response from OpenAI service for conversation {conversation_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Invalid response from AI service"
            )

        # Add assistant response and save
        existing_conversation["conversation"].append({"role": "assistant", "content": assistant_message})
        
        try:
            r.set(conversation_id, json.dumps(existing_conversation))
            logger.info(f"Successfully saved conversation {conversation_id} with {len(existing_conversation['conversation'])} messages")
        except redis.RedisError as e:
            logger.error(f"Error saving conversation {conversation_id}: {e}")
            # Still return the conversation even if saving fails
            logger.warning(f"Conversation {conversation_id} processed but not saved to Redis")

        return existing_conversation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing conversation {conversation_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )