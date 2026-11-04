# pydantic (data validation library)
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import  Optional, Literal
from pydantic import BaseModel, EmailStr,Field

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = "gemini-3.8-flash",
    temperature = 1,
    max_output_tokens = 25000,
    thinking_budget = 25000
)

class   Review(BaseModel):
    key_themes: list[str]= Field(description='Write down all key themes in this review')
    summary: str = Field(description='Write down summary of this review')
    sentiment: Literal['positive', 'negative', 'neutral'] = Field(description='Write down sentiment of this review')
    pros: list[str]= Field(description='Write down proportions of this review')
    cons: list[str]= Field(description='Write down conservation of this review')
    rating: Optional[float] = Field(description='Write down rating of the product')

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

#result.model_dump() → converts the entire object into a Python dictionary.
print('result',result)
print('rating',result.rating)