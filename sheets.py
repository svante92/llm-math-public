import os
from datetime import datetime
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
def append_data_to_sheet(problem: str):
# Load credentials from environment variables
    credentials = Credentials.from_service_account_info({
        "type": "service_account",
        "project_id": os.getenv("GOOGLE_PROJECT_ID"),
        "private_key_id": os.getenv("GOOGLE_PRIVATE_KEY_ID"),
        "private_key": os.getenv("GOOGLE_PRIVATE_KEY").replace('\\n', '\n'),
        "client_email": os.getenv("GOOGLE_CLIENT_EMAIL"),
        "client_id": os.getenv("GOOGLE_CLIENT_ID"),
        "auth_uri": os.getenv("GOOGLE_AUTH_URI"),
        "token_uri": os.getenv("GOOGLE_TOKEN_URI"),
        "auth_provider_x509_cert_url": os.getenv("GOOGLE_AUTH_PROVIDER_CERT_URL"),
        "client_x509_cert_url": os.getenv("GOOGLE_CLIENT_CERT_URL")
    })
        # The ID of the spreadsheet
    SPREADSHEET_ID = '1L_Uhxz3zNBtyCGsvMIcRI-X8OmRmXXKxb905Yrq5z4Y'
    RANGE_NAME = 'Sheet1!A:B'  # Access every row in columns A and B
        # Build the service
    service = build('sheets', 'v4', credentials=credentials)
        # Get the current date and time
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Prepare the values to append
    values = [
        [current_datetime, problem]
    ]
    body = {
        'values': values
    }
        # Append the values to the sheet
    result = service.spreadsheets().values().append(
        spreadsheetId=SPREADSHEET_ID, range=RANGE_NAME,
        valueInputOption="RAW", body=body,
        insertDataOption="INSERT_ROWS").execute()
    print(f"{result.get('updates').get('updatedCells')} cells updated.")

