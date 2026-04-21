import requests
import sys

API_KEY = "AIzaSyB0oMay-Mkh4mrpTY_Db1XFvAsJCIUNExo"
models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp", "gemini-2.5-flash"]

for model in models:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={API_KEY}"
    payload = {"contents": [{"parts": [{"text": "Say 'hello'"}]}]}
    print(f"Testing {model}...")
    try:
        res = requests.post(url, json=payload)
        if res.status_code == 200:
            print("  [SUCCESS] Status 200")
        else:
            print(f"  [FAILED] Status {res.status_code}: {res.text}")
    except Exception as e:
        print(f"  [ERROR] {e}")
