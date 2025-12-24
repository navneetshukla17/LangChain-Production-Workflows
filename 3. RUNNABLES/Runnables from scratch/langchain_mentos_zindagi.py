import random
from abc import ABC, abstractmethod

class Runnable(ABC):
    @abstractmethod
    def invoke(self, input_dict):
        pass

class NakliLLM:
    def __init__(self):
        print("LLM created & Ready to serve. \n")
    
    def invoke(self, input_dict):
        response_list = [
            'Virat kohli is the best batsman in the world',
            'Attack on Titan dekho bhai',
            'Game of thrones is GOAT',
            'AI stands for Artificial Intelligence',
            'Modiji Zinda Baad !'
        ]
        print('This class is deprecated in latest versions..')
        return {'response': random.choices(response_list)}
    

    def predict(self, prompt):
        response_list = [
            'Virat kohli is the best batsman in the world',
            'Attack on Titan dekho bhai',
            'Game of thrones is GOAT',
            'AI stands for Artificial Intelligence',
            'Modiji Zinda Baad !'
        ]
        print('This class is deprecated in latest versions..')
        return {'response': random.choices(response_list)}


class NakliPromptTemplate(Runnable):
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables
    
    def invoke(self, input_dict):
        return self.template.format(**input_dict)
    
    def format(self, input_dict):
        print('This class is deprecated in latest versions..')
        return self.template.format(**input_dict)


class NakliLLMChain:
    def __init__(self, llm, prompt):
        self.llm = llm
        self.prompt = prompt
    def invoke(self, input_dict):
        final_prompt = self.prompt.format(input_dict)
        result = self.llm.predict(final_prompt)
        return result['response']
    
    def run(self, input_dict):
        final_prompt = self.prompt.format(input_dict)
        result = self.llm.predict(final_prompt)
        print('This class is deprecated in latest versions..')
        return result['response']



llm = NakliLLM()

prompt = NakliPromptTemplate(
    template='what is the {topic}',
    input_variables=['topic']
)

chain = NakliLLMChain(
    llm=llm,
    prompt=prompt
)

print(chain.run({'topic': 'cricket'}))










