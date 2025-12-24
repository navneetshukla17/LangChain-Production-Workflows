from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv

# Load gemini api key
load_dotenv()

# Webpage
url = 'https://www.amazon.in/Apple-MacBook-Laptop-14%E2%80%91core-20%E2%80%91core/dp/B0DLHTDZVL/ref=sr_1_3?crid=ZVISWZL433OU&dib=eyJ2IjoiMSJ9.NXG4biOkzXZ8ec2Ng71So-0s9RUSHCHBzz007-74Hvs-wS0Pn9yd1t8Z5Jhnq0YUycWpVKoOk3BgccQ0_i0r7Atkrz5qI2jQho42DzR1MVT5o-TCQzivmfksOjuni2vP2ilRiJ0IBbNqQteIhu3Fgiv5qVjVdZVE8F5HLiLh2hAfsdBAfDNypS8JHf7jJuePKojyFcxOdfPC4sg82zGehvOpqx2LXHhV_0VSAe8ndCc.tfUQX-3QfOF7U8XavuYLXSCkoztuzqc5sr8Ov-cBn9w&dib_tag=se&keywords=macbook+pro+m4&qid=1766146230&sprefix=mackbook+pro+m4%2Caps%2C189&sr=8-3'
loader = WebBaseLoader(web_path=url)
webpage = loader.load()[0].page_content.split()

# Model
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')

# Prompt
prompt = PromptTemplate(
    template='Tell me is should buy this product or not by looking at the review -{text}',
    input_variables=['text']
)

# Output Parser
parser = StrOutputParser()

# Runnable/LCEL
chain = prompt | model | parser 

# Trigger LCEL
result = chain.invoke({'text': webpage})

# Final Output
print("My Decision Is: ", result)