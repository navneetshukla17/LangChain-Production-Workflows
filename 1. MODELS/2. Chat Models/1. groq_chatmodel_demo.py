from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load groq api key
load_dotenv()

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=50
)
result = model.invoke("Poem on rain in 5 lines")
print(result.content)