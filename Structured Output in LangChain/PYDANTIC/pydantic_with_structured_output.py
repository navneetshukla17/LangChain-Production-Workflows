from typing import TypedDict, Optional, Literal
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load the api key
load_dotenv()

# Model configuration
model = ChatGroq(model='llama-3.1-8b-instant')
# model = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')

# Import reviews
with open('review.txt', 'r', encoding='utf-8') as f:
    review_text = f.readlines()


class Review(BaseModel):
    key_points: list[str] = Field(description='Write all the keypoints from the review.')
    summary: str = Field(description='Write a concise summary of the review.')
    sentiment: Literal['pos', 'neg'] = Field(description='Write the sentiment of the review')
    pros: Optional[list[str]] = Field(default=None, description='Write advantages from the review.')
    cons: Optional[list[str]] = Field(default=None, description='Write disadvantages from the review.')
    name: Optional[str] = Field(default=None, description='Write reviewers name from the review.')


structured_model = model.with_structured_output(Review)
result = structured_model.invoke(review_text)
print(result)