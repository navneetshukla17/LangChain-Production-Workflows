from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

# Load huggingface inference api key
load_dotenv()

# Model congfiguration
llm = HuggingFaceEndpoint(
    repo_id='Qwen/Qwen2.5-7B-Instruct',
    task='text-generation'   
)
model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India ?")
print(result.content)



