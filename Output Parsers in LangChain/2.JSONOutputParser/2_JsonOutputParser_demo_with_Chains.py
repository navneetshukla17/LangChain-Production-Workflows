from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
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

# Output parser
parser = JsonOutputParser()

# prompt template
template = PromptTemplate(
    template='Give 5 facts about {topic} {format_instructions}',
    input_variables=['topic'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

# Chaining
chain = template | model | parser

result = chain.invoke({'topic': 'black hole'})
print(result)




