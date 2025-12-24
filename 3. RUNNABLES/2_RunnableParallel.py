from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import RunnableParallel, RunnableSequence
from dotenv import load_dotenv

# Load gemini api key
load_dotenv()

# Model
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')

# Prompts
prompt1 = PromptTemplate(
    template='Tell me a joke on - {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='Explain me this joke from the following - {topic}',
    input_variables=['topic']
)

# Output Parser
parser = StrOutputParser()

# using | operator
# parallel_chain = RunnableParallel({
#     'joke': prompt1 | model | parser,
#     'explain': prompt2 | model |parser
# })

# Runnable
parallel_chain = RunnableParallel({
    'joke': RunnableSequence(prompt1, model, parser),
    'explain': RunnableSequence(prompt2, model, parser)
})

# Trigger Runnables
result = parallel_chain.invoke({'topic': 'India'})

# Final Output
print(f"Joke: {result['joke']}")
print()
print(f"Explaination: {result['explain']}")