import requests
import os
from dotenv import load_dotenv
from io import BytesIO

# Load environment variables from .env file
load_dotenv()
APP_ID = os.getenv("WOLFRAM_APP_ID")  # Ensure this is set in your .env file
BASE_URL = "http://api.wolframalpha.com/v2/query"

def generate_graph_from_query(query: str) -> BytesIO:
    """
    Generate a graph image from a natural language query using Wolfram Alpha API.

    Args:
        query (str): The natural language query to generate the graph.

    Returns:
        BytesIO: The image data as a byte stream.
    """
    try:
        # Parameters for the API request
        params = {
            "input": query,
            "appid": APP_ID,
            "output": "JSON"  # Requesting JSON response for easier parsing
        }
        
        # Make the request to Wolfram Alpha
        response = requests.get(BASE_URL, params=params)
        
        if response.status_code == 200:
            # Parse JSON response
            data = response.json()
            
            # Extract the first plot pod and find the image URL
            pods = data.get("queryresult", {}).get("pods", [])
            for pod in pods:
                if "plot" in pod.get("title", "").lower():
                    subpods = pod.get("subpods", [])
                    for subpod in subpods:
                        img_url = subpod.get("img", {}).get("src")
                        if img_url:
                            # Download the image and return as BytesIO
                            img_response = requests.get(img_url)
                            if img_response.status_code == 200:
                                return BytesIO(img_response.content)
            raise Exception("No graph image found for the query.")
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")
    except Exception as e:
        raise Exception(f"Failed to generate graph: {str(e)}")

    

