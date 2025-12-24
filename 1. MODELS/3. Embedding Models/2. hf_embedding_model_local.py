from langchain_huggingface import HuggingFaceEmbeddings

# Model configuration
embedding_model_local = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)

# Text to embed
text = "Delhi is the capital of India"

# Document to embed
document = """
# Introduction to Artificial Intelligence
Artificial Intelligence has transformed the way we interact with technology. From virtual assistants to recommendation systems, AI is everywhere in our daily lives.
## Key Applications
Machine learning algorithms power modern applications across various industries. Healthcare uses AI for diagnostic imaging and drug discovery. Financial institutions employ AI for fraud detection and risk assessment. Retail businesses leverage AI to personalize customer experiences and optimize inventory management.
## The Future
As AI technology continues to evolve, we can expect even more innovative applications. The integration of AI with other emerging technologies like quantum computing and robotics promises to unlock new possibilities we haven't yet imagined.
The journey of AI is just beginning, and its potential to solve complex problems remains boundless.
"""

result = embedding_model_local.embed_documents(document)
print(str(result))
