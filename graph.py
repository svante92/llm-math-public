import requests
import os
from dotenv import load_dotenv
from io import BytesIO

# Load environment variables from .env file
load_dotenv()
APP_ID = os.getenv("WOLFRAM_APP_ID")  # Ensure this is set in your .env file
BASE_URL = "https://api.wolframalpha.com/v1/simple"

def generate_graph_from_query(query: str) -> BytesIO:
    """
    Generate a graph image from a natural language query using Wolfram Alpha API.

    Args:
        query (str): The natural language query to generate the graph.

    Returns:
        BytesIO: The image data as a byte stream.
    """
    try:
        # Make the API request
        response = requests.get(BASE_URL, params={"appid": APP_ID, "input": query})

        # Check the response
        if response.status_code == 200:
            # Return the image as a byte stream
            return BytesIO(response.content)
        
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")
    except Exception as e:
        raise Exception(f"Failed to generate graph: {str(e)}")

    

