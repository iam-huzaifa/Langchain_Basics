from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
load_dotenv(verbose=True)

model = ChatDeepSeek(model="deepseek-chat")

print("2. Model initialized")
result = model.invoke('give me a best coding tip that helps in industry.')

print("3. API response received")
print("Response:", repr(result.content))
print("Full result:", result)