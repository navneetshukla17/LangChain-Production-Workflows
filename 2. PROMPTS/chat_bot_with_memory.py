from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id='Qwen/Qwen2.5-7B-Instruct')

model = ChatHuggingFace(llm=llm)

# storing chat history for LLMs memory.
chat_history = []

while True:
    user_input = input("You: ")
    chat_history.append(user_input) # saving user message (Human Message).
    if user_input == 'exit':
        break
    response = model.invoke(chat_history) # sending the whole chat history instead of sending one single message
    chat_history.append(response) # saving the response given by the AI Model (AI Message).
    print(f"JARVIS: {response.content}")


print(f"Chat History:\n {chat_history}")