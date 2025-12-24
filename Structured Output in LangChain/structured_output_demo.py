from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated, Optional, Literal
from dotenv import load_dotenv

# Load api key
load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')

# Import reviews
with open('review.txt', 'r', encoding='utf-8') as f:
    review_text = f.readlines()
    # print(review_text)

# Document formating schemas
class ReviewWithoutAnnotation(TypedDict): 
    summary: str
    sentiment: str

class ReviewWithAnnotation(TypedDict):
    summary: Annotated[str, "Return a brief summary of the given review"]
    sentiment: Annotated[str, "Return sentiment of the review either Positive, Negative or Neutral"]

class ReviewWithSpecificKeywords(TypedDict):
    key_points: Annotated[list[str], "Return all the key points mentioned in the review"]

class ReviewWithProandCons(TypedDict):
    pros: Annotated[Optional[list[str]], "Write all the advantages of the product inside a list"]
    cons: Annotated[Optional[list[str]], "Write all the disadvantages of the product inside a list"]

class ReviewSentiment(TypedDict):
    name: Annotated[Optional[str], "Write the persons name in the review"]
    sentiment: Annotated[Literal['pos', 'neg'], 'wirte pos if the sentiment is positive']


# Run the model
structured_model = model.with_structured_output(ReviewSentiment) # Output structure schema


result = structured_model.invoke(review_text)
# print(f"Persons name & sentiment: {result}") # Issue: Returning products name instead of reviewrs name.
# print(f"Pros: {result['pros']}")
# print(f"Cons: {result['cons']}")
# print(f"Main Highlights: {result['key_points']}")
# print(f"Summary:\n {result['summary']}")
# print(f"Sentiment: {result['sentiment']}")






