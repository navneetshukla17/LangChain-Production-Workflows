from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# sample text
text = """
Space exploration has led to incredible scientific discoveries. From landing on the Moon to exploring Mars, humanity continues to push the boundaries of what’s possible beyond our planet.

These missions have not only expanded our knowledge of the universe but have also contributed to advancements in technology here on Earth. Satellite communications, GPS, and even certain medical imaging techniques trace their roots back to innovations driven by space programs.
"""
# Loader pdf
loader = PyPDFLoader(file_path='dl-curriculum.pdf')

# sample pdf
document = loader.load()

# text/document splitter
splitter1 = TokenTextSplitter(
    chunk_size=500,
    chunk_overlap=200,
)

splitter2 = TokenTextSplitter(
    chunk_size=500,
    chunk_overlap=200,
    separators=[
        "\nA. ",
        "\nB. ",
        "\nC. ",
        "\n1. "
        "\n2. "
        "\n3. "
        "\n4. "
        "\n5. "
        "\n6. "
        "\n7. "
        "\n8. "
    ]
)
  
result1 = splitter1.split_text(text=text)

chunks = splitter1.split_documents(documents=document)

for i, chunk in enumerate(chunks):
    print(f"\n --- Chunk: {i}---")
    print(chunk.page_content)