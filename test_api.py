import requests
import sys

API_KEY = "AIzaSyB0oMay-Mkh4mrpTY_Db1XFvAsJCIUNExo"
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"

payload = {
    "contents": [{"parts": [{"text": "Say 'hello world'"}]}]
}

try:
    response = requests.post(url, json=payload)
    print("Status:", response.status_code)
    print("Response:", response.text)
except Exception as e:
    print("Error:", e)
