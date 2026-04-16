import os
import ulid
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

load_dotenv()

# --- FIXING LANGFUSE ENVIRONMENT VARIABLES ---
# The @observe() decorator and CallbackHandler rely on specific environment variables.
# In your .env, the keys are swapped and mislabeled. We fix them here programmatically.

# 1. Extract values based on their actual content (pk- is public, sk- is secret)
raw_public = os.getenv("LANGFUSE_PRIVATE_KEY") # This contains 'pk-lf-...'
raw_secret = os.getenv("LANGFUSE_PUBLIC_KEY")  # This contains 'sk-lf-...'

# 2. Re-set them to the standard names Langfuse expects
os.environ["LANGFUSE_PUBLIC_KEY"] = raw_public
os.environ["LANGFUSE_SECRET_KEY"] = raw_secret
os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse")

# Initialize the global client with corrected keys
langfuse_client = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host=os.environ["LANGFUSE_HOST"]
)

def get_llm(model_name="google/gemini-2.0-flash-001"):
    api_key = os.getenv("OPENROUTER_API_KEY_1_2_3")
    return ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://github.com/antoine-lopez/hackathon-reply",
            "X-Title": "Fraud Detection Agent System",
        }
    )

def generate_session_id():
    """Generate a unique session ID using TEAM_NAME and ULID."""
    team_name = os.getenv('TEAM_NAME', 'Antoine-Lopez-Team')
    return f"{team_name}-{ulid.new().str}"

def get_langfuse_handler():
    # The handler will now automatically pick up the corrected environment variables
    return CallbackHandler()
