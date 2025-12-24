from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import CSVLoader
from dotenv import load_dotenv

# Load hugginface api key
load_dotenv()

# CSV file
loader = CSVLoader(file_path='Social_Network_Ads.csv', encoding='utf-8')
csv_file = loader.load()

# Convert csv into text
csv_text = "\n".join(doc.page_content for doc in csv_file)

# LLM config
# HuggingFaceH4/zephyr-7b-beta - optional LLM model
llm = HuggingFaceEndpoint(
    repo_id='HuggingFaceH4/zephyr-7b-beta',
    task='text-generation'
)
# Model config
model = ChatHuggingFace(llm=llm)

# Prompt
prompt = PromptTemplate(
    template="""
what is the maximum amount in the following CSV data:

{file}
""",
    input_variables=['file']
)

# Output Parser
parser = StrOutputParser()

# Runnable/LCEL
chain = prompt | model | parser

# Trigger Runnable
result = chain.invoke({'file': csv_text})

# Final Output
print(result)


# ================== Model is Hallucinating ========================