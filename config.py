
import os
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL  = "text-embedding-3-small"
LLM_MODEL        = "llama3-8b-8192"
LLM_TEMPERATURE  = 0
PDF_DIR          = "pdfs"
VECTORSTORE_DIR  = "vectorstore"
CHUNK_SIZE       = 1000
CHUNK_OVERLAP    = 200
TOP_K_RESULTS    = 5
SYSTEM_PROMPT    = """You are a helpful academic tutor.
Use ONLY the provided context to answer the question.
If the answer is not found, say: I dont have enough information."""

