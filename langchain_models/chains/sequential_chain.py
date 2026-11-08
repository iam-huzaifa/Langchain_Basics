
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=1.0,
    max_output_tokens=25000
)

class Summary(BaseModel):
    summary: str = Field(
        description="Give a summary of the game"
    )

class Game(BaseModel):
    tips: list[str] = Field(
        description="Give tips for beginners to play the game"
    )

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a gaming expert. Give a short summary of the game."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Game name: {game_name}")
])

game_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a gaming coach. Give beginner tips."),
    ("human", "Game summary: {summary}")
])

summary_model = model.with_structured_output(Summary)

game_model = model.with_structured_output(Game)


base_chain = (
    summary_prompt
    | summary_model
    | (lambda x: {"summary": x.summary})
    | game_prompt
    | game_model
    | (lambda x: {
        "game": x,
        "tip": "\n".join(x.tips)
    })
)


memory = {}

def get_session_history(session_id):

    if session_id not in memory:
        memory[session_id] = InMemoryChatMessageHistory()

    return memory[session_id]

chain = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="game_name",
    history_messages_key="chat_history",
    output_messages_key="tip"
)

config = {
    "configurable": {
        "session_id": "user1"
    }
}

result = chain.invoke(
    {"game_name": "Foosball"},
    config=config
)

print("\nFirst interaction:")
print(result["game"].tips)

# 10. Second interaction (same session)
result = chain.invoke(
    {"game_name": "Foosball"},
    config=config
)

print("\nSecond interaction:")
print(result["game"].tips)


print("\nConversation history:")
for message in memory["user1"].messages:
    print(message.type, ":", message.content)


print("\nFinal result:")
print(result)

print("\nFinal tips:")
print(result["tip"])