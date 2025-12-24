#               SYSTEM ARCHITECTURE FOR SIMPLE CHAIN:
# ┌─────────────────────────────────────────────────────────────────┐
# │                         USER INTERFACE                          │
# └─────────────────────────────────────────────────────────────────┘
#                                 │
#                                 ▼
#                     ┌───────────────────────┐
#                     │   STEP 1: Input       │
#                     │   User enters prompt  │
#                     └───────────────────────┘
#                                 │
#                                 ▼
#                     ┌───────────────────────┐
#                     │   STEP 2: Processing  │
#                     │   ┌─────────────────┐ │
#                     │   │      LLM        │ │
#                     │   │   (Generate     │ │
#                     │   │    Response)    │ │
#                     │   └─────────────────┘ │
#                     └───────────────────────┘
#                                 │
#                                 ▼
#                     ┌───────────────────────┐
#                     │   STEP 3: Output      │
#                     │   Display LLM         │
#                     │   response to user    │
#                     └───────────────────────┘
#                                 │
#                                 ▼
# ┌─────────────────────────────────────────────────────────────────┐
# │                         USER INTERFACE                          │
# │                    (Response Display)                           │
# └─────────────────────────────────────────────────────────────────┘
# Flow Summary: User Prompt → LLM Processing → Response Output

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
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

# Model config
model = ChatHuggingFace(llm=llm)

# Design Prompt template
prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

# Output parser
parser = StrOutputParser()

# Simple chain using LCEL (LangChain Expression Language)
chain = prompt | model | parser

# Trigger chain
result = chain.invoke({'topic': 'Generative AI'})

# Final output (to the user)
print(result)

# Visualize 
print("Visual Representation of Simple Chain:")
chain.get_graph().print_ascii()