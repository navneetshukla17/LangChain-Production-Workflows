from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

# Load the api key.
load_dotenv()

llm = HuggingFaceEndpoint(repo_id='Qwen/Qwen2.5-7B-Instruct')
model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content='You are an helpful AI assistant'),
    HumanMessage(content='what is the capital of America ?')
]

response = model.invoke(messages)
messages.append(AIMessage(content=response.content))


print(messages)





