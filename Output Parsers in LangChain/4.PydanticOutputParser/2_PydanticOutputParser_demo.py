# Task: extract (name, age, city) from LLM outpus in JSON format
# Condition: age must be integer and (18 < age < 90)

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load api key
load_dotenv()

# LLM config
llm = HuggingFaceEndpoint(
    repo_id='google/gemma-2-2b-it',
    task='text-generation'
)

# Model config
model = ChatHuggingFace(llm=llm)

# Pydantic output parser schema
class Person(BaseModel):
    name: str = Field(description='Name of the person'),
    age: int = Field(gt=18, lt=90, description='Age of the person'),
    city: str = Field(description='Name of the city from the person belongs')

# Output parser object
parser = PydanticOutputParser(pydantic_object=Person)


# Prompt template
template = PromptTemplate(
    template='Generate name, age, city of a fictional {place} character {format_instructions}',
    input_variables=['place'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)


# Chaining
chain = template | model | parser

result = chain.invoke({'place': 'Dubai'})

print(result)


