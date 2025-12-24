from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

# Output parser
parser = StrOutputParser()

# Chaining
chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({'topic': 'black hole'})

# Final output
print(f"5 line summary on black hole: \n{result}")




