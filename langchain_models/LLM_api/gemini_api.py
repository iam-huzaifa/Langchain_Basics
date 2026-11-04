from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.7,
    max_output_tokens=25000,
    thinking_budget=25000
)

result = model.invoke(
    "Give me one coding tip that helps in industry. "
    "Explain it in 3 sentences with an example."
)

print("ANSWER:")
print(result.content)

print("\nMETADATA:")
print(result.response_metadata)