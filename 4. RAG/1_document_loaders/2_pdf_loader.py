from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

# Load gemini api key
load_dotenv()

# PDF
loader = PyPDFLoader(file_path='dl-curriculum.pdf')
pdf = loader.load()
print(pdf[0].page_content)

# Model
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')

# Prompt
# prompt = PromptTemplate(
#     template='',
#     input_variables=['']
# )

# Output Parser
parser = StrOutputParser()

# Runnables/LCEL
# chain = prompt | model | parser

# Trigger Runnable
# chain.invoke({'': ''})

# Final Output