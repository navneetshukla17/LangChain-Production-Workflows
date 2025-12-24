from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import RunnableSequence
from dotenv import load_dotenv

# Load huggingface api key
load_dotenv()

# LLM config
llm = HuggingFaceEndpoint(
    repo_id='google/gemma-2-2b-it',
    task='text-generation'
)

# Model config
model = ChatHuggingFace(llm=llm)

# Prompts
prompt1 = PromptTemplate(
    template='Tell me a joke on {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='Explain me the joke from this {text}',
    input_variables=['text']
)

# Output Parser
parser = StrOutputParser()

# using LCEL
# chain = prompt | model | parser 

# two prompt chain
# chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

chain = RunnableSequence(prompt1, model, parser)

print(chain.invoke({'topic': 'AI'}))