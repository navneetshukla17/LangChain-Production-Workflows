from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import RunnablePassthrough, RunnableParallel, RunnableSequence
from dotenv import load_dotenv

# Load groq api key
load_dotenv()

# Model
model = ChatGroq(model='llama-3.1-8b-instant')

# Prompts
prompt1 = PromptTemplate(
    template='Write a joke on {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='Explain the give {topic}',
    input_variables=['topic']
)

# Output Parser
parser = StrOutputParser()

# Runnables
# sequential_chain = prompt1 | model | parser
sequential_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explain': RunnableSequence(prompt2, model, parser)
})

# combined_chain = sequential_chain | parallel_chain
combined_chain = RunnableSequence(sequential_chain, parallel_chain)

# Trigger Runnables
result = combined_chain.invoke({'topic': 'America'})

# Final Output
print(result)
