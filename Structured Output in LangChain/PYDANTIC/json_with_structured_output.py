# Necessary libraries
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load api key
load_dotenv()

# Model configuration
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')

# Import reviews
with open('review.txt', 'r', encoding='utf-8') as f:
    review_text = f.readlines()

# Import output schema
with open('schema.json', 'r', encoding='utf-8') as f:
    json_schema = json.load(f)

structured_model = model.with_structured_output(json_schema)
result = structured_model.invoke(review_text)
print(result)

