# ============================= Overall Workflow ================================
# Step 1: Take feedback(prompt) from the customers send it to the Model-1
# Step 2(Model-1): Generate sentiment analysis on that feedback(prompt)
# Step 3(Model-2): If sentiment is positive: Give feedback form to the user with a thankyou message
# Step 4(Model-3): If sentiment is negative: Write mail to customer support to assist the users problems.

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace # Model-1
from langchain_google_genai import ChatGoogleGenerativeAI # Model-2
from langchain_groq import ChatGroq # Model-3
from langchain_core.prompts import PromptTemplate
from langchain_classic.schema.runnable import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

# Load api key
load_dotenv()

# ============================= feedback: document ================================
with open('document.txt', 'r', encoding='utf-8') as f:
    document = f.readlines()

# ============================= Models ================================
# LLM config: Model-1
llm = HuggingFaceEndpoint(
    repo_id='google/gemma-2-2b-it',
    task='text-generation'
)
model1 = ChatHuggingFace(llm=llm)

model2 = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')

model3 = ChatGroq(model='llama-3.1-8b-instant')

# ============================= Output Parser ================================
parser = StrOutputParser()

# Enforce consistent output 
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="Classify the sentiment of the feedback text.")


parser2 = PydanticOutputParser(pydantic_object=Feedback)

# ============================= Prompt templates =============================
prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into positive or negative. {feedback} {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)
prompt2 = PromptTemplate(
    template='Write an appropriate feedback for this positive feedback {feedback}',
    input_variables=['feedback']
)
prompt3 = PromptTemplate(
    template='Write an appropriate feedback for this negative feedback {feedback}',
    input_variables=['feedback']
)

# ============================= Chains =============================

classifier_chain = prompt1 | model1 | parser2

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model2 | parser),
    (lambda x: x.sentiment == 'negative', prompt3 | model3 | parser),
    RunnableLambda(lambda x: "Could not find the sentiment !")
)
# Combined chain: classifier_chain + branch_chain
final_chain = classifier_chain | branch_chain

# Trigger chain
result = final_chain.invoke({'feedback': 'This is beautiful smartphone.'})
print(result)

# ============================= Visualize the branch chain =============================
final_chain.get_graph().print_ascii()