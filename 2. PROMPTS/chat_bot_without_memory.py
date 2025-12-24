from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id='Qwen/Qwen2.5-7B-Instruct')

model = ChatHuggingFace(llm=llm)


while True:
    user_input = input("You: ")
    if user_input == 'exit':
        break
    response = model.invoke(user_input)
    print(f"JARVIS: {response.content}")


print(f"Chat History:\n {user_input}")