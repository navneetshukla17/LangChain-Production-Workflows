from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load huggingface api key
load_dotenv()

# Document
loader = DirectoryLoader(
    path='Books',
    glob='*.pdf',
    loader_cls=PyPDFLoader,
)

doc = loader.load()
print(doc)

# docs = loader.lazy_load()
# for doc in docs:
#     print(doc.page_content)


# # LLM config
# llm = HuggingFaceEndpoint(
#     repo_id='HuggingFaceH4/zephyr-7b-beta',
#     task='text-generation'
# )
# # Model config
# model = ChatHuggingFace(llm=llm)

# # Prompt
# prompt = PromptTemplate(
#     template='summarise the give {document} in 100 characters.',
#     input_variables=['document']
# )

# # Output Parser
# parser = StrOutputParser()

# # Runnable/LCEL
# chain = prompt | model | parser

# # Trigger Runnable
# result = chain.invoke({'document': doc})

# # Final Output
# print("Document Summary:\n", result)

