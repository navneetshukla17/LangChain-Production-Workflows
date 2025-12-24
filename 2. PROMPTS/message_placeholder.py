from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from dotenv import load_dotenv

# Load api key
load_dotenv()

llm = HuggingFaceEndpoint(repo_id='Qwen/Qwen2.5-7B-Instruct')
model = ChatHuggingFace(llm=llm)

# chat template
chat_template = ChatPromptTemplate([
    ('system', 'you are an customer support staff'),
    MessagesPlaceholder(variable_name='chat_history'), # Adding previous chat history
    ('human', '{query}')
])

# load chat history
chat_history = []

with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

# crete prompt
prompt = chat_template.invoke({'chat_history': chat_history, 'query': 'Where is my refund'})
print(prompt)
