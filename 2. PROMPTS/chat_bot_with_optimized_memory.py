from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

# Load the api key
load_dotenv()

llm = HuggingFaceEndpoint(repo_id='Qwen/Qwen2.5-7B-Instruct')

model = ChatHuggingFace(llm=llm)

# storing chat history for LLMs memory.

chat_history = [
    SystemMessage(content='You are an helpfull AI assistant'),
    HumanMessage(content='what is the capital of India ?'),
]

while True:
    user_input = input("You: ")
    chat_history.append(user_input) # saving user message (Human Message).
    if user_input == 'exit':
        break
    response = model.invoke(chat_history) # sending the whole chat history instead of sending one single message
    chat_history.append(AIMessage(content=response.content)) # saving the response given by the AI Model (AI Message).
    print(f"JARVIS: {response.content}")


print(f"Chat History:\n {chat_history}")