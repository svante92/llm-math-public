import os
from dotenv import load_dotenv

def load_environment_variables():
    load_dotenv()
    return os.getenv("OPENAI_API_KEY") 