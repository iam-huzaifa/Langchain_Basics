from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated,Optional

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = "gemini-3.8-flash",
    temperature = 1,
    max_output_tokens = 25000,
    thinking_budget = 25000
)

class   Review(TypedDict):
    key_themes: Annotated[list[str], 'Write down all key themes in this review']
    summary: Annotated[list[str], 'Write down summaries of review']
    sentiment: Annotated[list[str], 'Write down sentiments of review']
    pros: Annotated[list[str], 'Write down proportions of reviews']
    cons: Annotated[list[str], 'Write down consensus of reviews']
    name: Annotated[Optional[list[str]], 'Write down names of review']
    rating: Annotated[Optional[float], 'Write down rating of review']

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""Hi, my name is Huzaifa, and I have been using the HP ProBook 640 G8. Overall, I think it is a good business laptop for students, programmers, and office users.

**Pros:**
- Good performance for programming, browsing, and everyday multitasking.
- Professional and simple design.
- Comfortable keyboard, especially for typing code and assignments.
- Useful for university work, online classes, and office tasks.

**Cons:**
- Not a good choice for heavy gaming or demanding graphics work.
- Battery life may not be great, especially if the laptop has an older or worn-out battery.
- Its hardware is older compared with newer laptop models.
- Display quality may feel average depending on the configuration.

**Final Recommendation:** Recommended for students and professionals who need a reliable laptop for everyday work, coding, and productivity. However, I would recommend checking the laptop's RAM, SSD, battery health, and price before buying it.""")

print('result',result)
print('structured_model',structured_model)
print('rating',result['rating'])