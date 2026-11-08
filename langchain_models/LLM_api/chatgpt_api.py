from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv(verbose=True)

print("1. Program started")
model = ChatOpenAI(model="gpt-6-luna",temperature = 1,max_completion_tokens=10)

print("2. Model initialized")
result = model.invoke('give me a best coding tip that helps in industry.')

print("3. API response received")
print("Response:", repr(result.content))
print("Full result:", result)