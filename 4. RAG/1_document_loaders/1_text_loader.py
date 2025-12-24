from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv

# Load huggingface api key
load_dotenv()

# Document
loader = TextLoader(file_path='cricket.txt', encoding='utf-8')
doc = loader.load()
# print(doc[0].page_content)

# LLM config
llm = HuggingFaceEndpoint(
    repo_id='google/gemma-2-2b-it',
    task='text-generation'
)
# Model config
model = ChatHuggingFace(llm=llm)

# Prompt
prompt = PromptTemplate(
    template='write a summary about {poem}',
    input_variables=['poem']
)

# Output Parser
parser = StrOutputParser()

# LCEL/Runnables
chain = prompt | model | parser

# Trigger Runnables
result = chain.invoke({'poem':doc[0].page_content})

# Final Output
final_result = """Poem Summary:\n{}""".format(result)
print(final_result)
