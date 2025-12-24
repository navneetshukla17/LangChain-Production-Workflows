# Step 1: Take input(prompt) from the user: Create Notes and Quizz from this Document
# Step 2: Parallel processing [Model1 - Create Notes & Model2 - Create Quizz] from the given Document
# Step 3: Model3 - Merege the output of both the Models and display it to the user

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import RunnableParallel
from dotenv import load_dotenv

# Load api key
load_dotenv()

# Load document
with open('document.txt', 'r', encoding='utf-8') as f:
    document = f.readlines()


# LLM config: for Model-1
llm = HuggingFaceEndpoint(
    repo_id='google/gemma-2-2b-it',
    task='text-generation'
)
# For Creating Notes
model1 = ChatHuggingFace(llm=llm)

# For Creating Quizz
model2 = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')

# For Merging Output of both Model1 & Model2
model3 = ChatGroq(model='llama-3.1-8b-instant')

# Prompt-1
prompt1 = PromptTemplate(
    template='Create Short and Simple Notes from the given {document}',
    input_variables=['document']
)
# Prompt-2
prompt2 = PromptTemplate(
    template='Create 5 Question Answers from the given {document}',
    input_variables=['document']
)
# Prompt-3
prompt3 = PromptTemplate(
    template='Merge the provided Notes and Quizz into a single document \n notes: {notes} quizz: {quizz}',
    input_variables=['notes', 'quizz']
)

# Output Parser
parser = StrOutputParser()

# Parallel Chain
parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quizz': prompt2 | model2 | parser
})
merge_chain = prompt3 | model3 | parser

# combined chain
final_chain = parallel_chain | merge_chain

# Trigger parallel chain
result = final_chain.invoke({'document': document})

print(result)

# Visualize the parallel chain
final_chain.get_graph().print_ascii()


