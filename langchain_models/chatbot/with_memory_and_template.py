# ChatPromptTemplate
# MessagesPlaceholder

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage
from dotenv import load_dotenv
load_dotenv()

# 1. Initialize Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=1,
    max_output_tokens=25000
)

# 2. Create Chat Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a tech industry expert in Pakistan. "
               "Give simple and practical answers. "
               "Consider previous conversation when answering."),

    MessagesPlaceholder(variable_name="chat_history"),

    ("human", "{question}")
])

# 3. Connect prompt with model
chain = prompt | model

# 4. Initialize conversation history
chat_history: list[BaseMessage] = []

print("Chatbot Started! Type 'exit' to stop.\n")

# 5. Chatbot loop
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot Closed!")
        break

    # 6. Invoke chain with current input and history
    result = chain.invoke({
        "question": user_input,
        "chat_history": chat_history
    })

    # 7. Save user and AI messages
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=result.content))

    # 8. Display AI response
    print("\nAI:", result.content)

    # 9. Display metadata
    print("\nMETADATA:")
    print("Response Metadata:", result.response_metadata)
    print("Usage Metadata:", result.usage_metadata)

    print("\n" + "-" * 50 + "\n")