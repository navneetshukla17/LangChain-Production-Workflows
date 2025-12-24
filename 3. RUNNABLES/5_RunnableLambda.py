from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import RunnableLambda, RunnableSequence, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

# Load huggingface api key
load_dotenv()

# LLM config
llm = HuggingFaceEndpoint(
    repo_id='google/gemma-2-2b-it',
    task='text-generation'
)

# Model
model = ChatHuggingFace(llm=llm)

# Prompts
prompt = PromptTemplate(
    template='Generate a joke on {topic}',
    input_variables=['topic']
)

# Output Parser
parser = StrOutputParser()

# word counter
def word_counter(text):
    return len(text.split())

word_counter_runnable = RunnableLambda(word_counter)

# Runnable
# using LCEL
# chain = prompt | model | parser

chain = RunnableSequence(prompt, model, parser)
parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': word_counter_runnable
})

# connect sequential and parallel chain
# using LCEL
#combined_chain = chain | parallel_chain

combined_chain = RunnableSequence(chain, parallel_chain)

# Trigger Runnable
result = combined_chain.invoke({'topic': 'cricket'})

final_result = """Joke: {}\nWord Count: {}""".format(result['joke'],result['word_count'])
# Final result
print(final_result)