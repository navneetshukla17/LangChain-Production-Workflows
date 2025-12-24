from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# sample python code to split
document = """

import random

class NakliLLM:
    def __init__(self):
        print('Nakli LLM Created...\n')
    
    def predict(self, prompt):
        response_list = [
            'Virat kohli is the best batsman in the world',
            'Attack on Titan dekho bhai',
            'Game of thrones is GOAT',
            'AI stands for Artificial Intelligence',
            'Modiji Zinda Baad !'
        ]
        return {'response':random.choices(response_list)}


class NakliPromptTemplate:
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables
    
    def format(self, input_dict):
        return self.template.format(**input_dict)


class NakliLLMChain:
    def __init__(self, llm, prompt):
        self.llm = llm
        self.prompt = prompt
    
    def run(self, input_dict):
        final_prompt = self.prompt.format(input_dict)
        result = self.llm.predict(final_prompt)

        return result['response']



# LLM config
llm = NakliLLM()

# Design prompt
prompt = NakliPromptTemplate(
    template='What is the capital of {country}',
    input_variables=['country']
)

chain = NakliLLMChain(llm=llm, prompt=prompt)
result = chain.run({'country':'India'})
print(result)
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=440,
    chunk_overlap=0
)

chunks = splitter.split_text(text=document)

for i, chunk in enumerate(chunks):
    print(f"\n---- Chunk: {i} ----\n")
    print(chunk)