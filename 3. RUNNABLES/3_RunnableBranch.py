from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import RunnableBranch, \
RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

# Load groq api key
load_dotenv()

# Model 
model = ChatGroq(model='llama-3.1-8b-instant')

# Prompts
prompt1 = PromptTemplate(
    template='Write a detailed report on the {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='Summarize the report {text}',
    input_variables=['text']
)

# Output Parser 
parser = StrOutputParser()

# Runnables
# LCEL
# report_gen_chain = prompt1 | model | parser

report_gen_chain = RunnableSequence(prompt1, model, parser)

# LCEL
# branch_chain = RunnableBranch(
#     (lambda x: len(x.split()) > 500, prompt2 | model | parser),
#     (lambda x: len(x.split()) < 500, RunnableLambda(lambda x: print('Generated text was greater than 500 words'))),
#     RunnablePassthrough()
# )

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 500, RunnableSequence(prompt2, model, parser)),
    (lambda x: len(x.split()) < 500, RunnableLambda(lambda x: print('Generated text was greater than 500 words'))),
    RunnablePassthrough()
)

# using LCEL
# final_chain = report_gen_chain | branch_chain

final_chain = RunnableSequence(report_gen_chain, branch_chain)

print(final_chain.invoke({'topic': 'Football'}))