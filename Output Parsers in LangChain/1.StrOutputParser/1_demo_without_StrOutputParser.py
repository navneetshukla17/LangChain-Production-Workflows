from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Load api key
load_dotenv()

# LLM config
llm = HuggingFaceEndpoint(
    repo_id='google/gemma-2-2b-it',
    task='text-generation'
)

# Model
model = ChatHuggingFace(llm=llm)

# 1st prompt: Detailed report on black hole.
template1 = PromptTemplate(
    template='Write detailed report on {topic}.',
    input_variables=['topic']
)
# 2nd prompt: 5 line summary on black hole.
template2 = PromptTemplate(
    template='Write 5 line summary on give {text}.',
    input_variables=['text']
)

# Execute prompt1
prompt1 = template1.invoke({'topic': 'black hole'})
res1 = model.invoke(prompt1)

# Execute prompt2
prompt2 = template2.invoke({'text': res1.content})
res2 = model.invoke(prompt2)

# Final output
print(res2.content)





