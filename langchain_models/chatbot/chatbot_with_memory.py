from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
load_dotenv(verbose=True)

print('1. Program Started')
model = ChatGoogleGenerativeAI(
    model = 'gemini-3.8-flash',
    temperature = 1.2,
    max_output_tokens = 25000,
    thinking_budget = 25000
)
chat_history = [
    SystemMessage(content = 'you are tech industrial expert in Pakistan')
]
while True:
    user_input = input('you: ')
    chat_history.append(HumanMessage(content = user_input))
    if user_input.lower() == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content = result.content))
    print('AI.', result.content)

    print("\nMETADATA:")
    print(result.response_metadata)
    print(result.usage_metadata)