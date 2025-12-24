#            SYSTEM ARCHITECTURE FOR SEQUENTIAL CHAIN:
# ┌─────────────────────────────────────────────────────────────────┐
# │                         USER INTERFACE                          │
# └─────────────────────────────────────────────────────────────────┘
#                                 │
#                                 ▼
#                     ┌───────────────────────┐
#                     │   STEP 1: Input       │
#                     │   User enters topic   │
#                     │   prompt: "{topic}"   │
#                     └───────────────────────┘
#                                 │
#                                 ▼
#                     ┌───────────────────────┐
#                     │   STEP 2: Generation  │
#                     │   ┌─────────────────┐ │
#                     │   │      LLM        │ │
#                     │   └─────────────────┘ │
#                     │   Instruction:        │
#                     │   "Generate detailed  │
#                     │   report on {topic}"  │
#                     └───────────────────────┘
#                                 │
#                                 ▼
#                     ┌───────────────────────┐
#                     │   Generated Report    │
#                     │   (Full detailed      │
#                     │   content)            │
#                     └───────────────────────┘
#                                 │
#                                 ▼
#                     ┌───────────────────────┐
#                     │   STEP 3: Extraction  │
#                     │   ┌─────────────────┐ │
#                     │   │      LLM        │ │
#                     │   └─────────────────┘ │
#                     │   Instruction:        │
#                     │   "Generate 5 key     │
#                     │   points from report" │
#                     └───────────────────────┘
#                                 │
#                                 ▼
#                     ┌───────────────────────┐
#                     │   STEP 4: Output      │
#                     │   Display 5 key       │
#                     │   points to user      │
#                     └───────────────────────┘
#                                 │
#                                 ▼
# ┌─────────────────────────────────────────────────────────────────┐
# │                         USER INTERFACE                          │
# │                    (Final Output Display)                       │
# └─────────────────────────────────────────────────────────────────┘
# Flow Summary: User Prompt1 → LLM Processing → Response Output → User Prompt2 (with generated response)
# → LLM Processing → Geneartes Final Response


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

# Prompt-1
prompt1 = PromptTemplate(
    template='Generate detailed report on {topic}',
    input_variables=['topic']
)

# Prompt-2
prompt2 = PromptTemplate(
    template='Generate 5 pointer summary from the {detailed report}',
    input_variables=['detailed report']
)
# parse final output
parser = StrOutputParser()

# Create chain: Sequential chain
chain = prompt1 | model | parser | prompt2 | model | parser
result = chain.invoke({'topic': 'Cricket'})

# Final output
print(result)

# Visualize 
print("Visual Representation of Sequential Chain:")
chain.get_graph().print_ascii()