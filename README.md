<div align="center">

# 🦜🔗 LangChain-Production-Workflows

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-00A67E?style=for-the-badge&logo=chainlink&logoColor=white)](https://python.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/navneet-shukla17/)

**Complete end-to-end LangChain implementation guide covering Models, Prompts, Output Parsers, Chains/LCEL, RAG pipelines, Memory, Tools, and Agents.**

*Production-ready examples with best practices for building LLM applications.*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Components Covered](#-components-covered)
- [Contributing](#-contributing)
- [Resources](#-resources)
- [License](#-license)
- [Support](#-support)
- [Acknowledgments](#-acknowledgments)
- [Contact](#-contact)

---

## 🎯 Overview

This repository provides complete, hands-on implementations of every major LangChain component. Whether you're building a simple chatbot or a complex multi-agent RAG system, you'll find production-ready code examples and best practices here.

### What makes this different?

- ✅ End-to-end workflows, not isolated snippets
- ✅ Real-world use cases with proper error handling
- ✅ Performance optimization techniques
- ✅ Local and cloud model examples
- ✅ Comprehensive documentation

---

## ✨ Features

### 🤖 Models

- **Vanilla LLMs** (OpenAI, Anthropic, Google)
- **Chat Models** (ChatOpenAI, ChatAnthropic)
- **Embedding Models** (OpenAI, HuggingFace, Cohere)
- **Local Models** (Ollama, LlamaCpp, GPT4All)
- **Open-Source LLMs** (Llama 2/3, Mistral, Falcon)

### 💬 Prompts & Memory

- Static & Dynamic Prompts
- PromptTemplate & ChatPromptTemplate
- Few-shot Learning Templates
- Chatbots with/without Memory
- Conversation Buffer/Summary/Window Memory

### 🔧 Output Parsers

- **StrOutputParser** - Simple string outputs
- **JsonOutputParser** - Structured JSON responses
- **PydanticOutputParser** - Type-safe parsing with validation

### ⛓️ Chains & LCEL (LangChain Expression Language)

- **RunnableSequence** - Sequential execution
- **RunnableParallel** - Parallel processing
- **RunnableBranch** - Conditional routing
- **RunnableLambda** - Custom functions
- **RunnablePassthrough** - Data passing

### 📚 RAG (Retrieval-Augmented Generation)

#### 1️⃣ Indexing Pipeline

**Data Ingestion:**
- `TextLoader` - Plain text files
- `PyPDFLoader` - PDF documents
- `DirectoryLoader` - Batch loading
- `WebBaseLoader` - Web scraping
- `GitHubLoader` - GitHub repositories
- `YoutubeLoader` - YouTube transcripts

**Text Splitting (Chunking):**
- Length-based splitting
- Text structure preservation (`\n\n`, `\n`, etc.)
- Document structure (code, markdown)
- Semantic meaning-based chunking

**Classes:**
- `CharacterTextSplitter` - Basic splitting
- `RecursiveCharacterTextSplitter` - Hierarchical splitting
- `SemanticChunker` - Meaning-aware chunking

**Vector Storage:**
- **Chroma DB** - Lightweight, embedded
- **FAISS** - Facebook AI Similarity Search
- **Pinecone** - Cloud-based, scalable

#### 2️⃣ Retrieval

**Data Source Retrievers:**
- `WikipediaRetriever`
- `ArxivRetriever`
- Vector store `.as_retriever()`

**Search Types:**
- Similarity Search
- **MMR** (Maximum Marginal Relevance)
- `MultiQueryRetriever`
- `ContextualCompressionRetriever`

#### 3️⃣ Augmentation & Generation

- Context injection (retrieved docs + user query)
- LLM generation with parametric + retrieved knowledge
- Hallucination reduction techniques

### 🛠️ Tools & Agents

- Custom tool creation
- Tool calling/function calling
- ReAct agents
- OpenAI Functions agents
- Multi-agent systems

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- API keys (OpenAI, Anthropic, etc.) - see `.env.example`

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/langchain-complete-guide.git
cd langchain-complete-guide

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Requirements

```txt
langchain>=0.1.0
langchain-community>=0.0.20
langchain-openai>=0.0.5
langchain-anthropic>=0.1.0
chromadb>=0.4.22
faiss-cpu>=1.7.4
pypdf>=4.0.0
beautifulsoup4>=4.12.0
youtube-transcript-api>=0.6.1
tiktoken>=0.5.2
pydantic>=2.5.0
python-dotenv>=1.0.0
```

---

## ⚡ Quick Start

*(Add quick start examples here)*

---

## 🧩 Components Covered

### Models

| Component | Description | Example |
|-----------|-------------|---------|
| **Vanilla LLMs** | Text completion | `HuggingFaceEndpoint(repo_id='google/gemma-2-2b-it', task='text-generation')` |
| **Chat Models** | Conversational | `ChatHuggingFace(model="google/gemma-2-2b-it")` |
| **Embeddings** | Vector representations | `HuggingFaceEmbeddings(model='sentence/transformers/all-MiniLM-L6-v2')` |
| **Local Models** | Run locally | `model='llama-3.1-8b-instant'` |

### RAG Components

| Stage | Tools | Purpose |
|-------|-------|---------|
| **Load** | TextLoader, PyPDFLoader | Data ingestion |
| **Split** | RecursiveCharacterTextSplitter | Chunking |
| **Embed** | OpenAIEmbeddings | Vector generation |
| **Store** | Chroma, FAISS, Pinecone | Persistence |
| **Retrieve** | MMR, Similarity | Context fetching |
| **Generate** | ChatOpenAI + Retrieved Docs | Answer synthesis |

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📚 Resources

### Official Documentation

- [LangChain Docs](https://python.langchain.com/)
- [LangChain API Reference](https://api.python.langchain.com/)
- [LangSmith (Observability)](https://smith.langchain.com/)

### Learning Materials

- [LangChain Cookbook](https://github.com/langchain-ai/langchain-cookbook)
- [RAG Best Practices](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)

### Related Projects

- [LlamaIndex](https://www.llamaindex.ai/) - Alternative RAG framework
- [Haystack](https://haystack.deepset.ai/) - NLP framework
- [LangGraph](https://langchain-ai.github.io/langgraph/) - Agent workflows

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## ⭐ Support

If you find this repository helpful, please consider:

- Giving it a ⭐ on GitHub
- Sharing it with your network
- Contributing improvements

---

## 🙏 Acknowledgments

- LangChain team for the amazing framework
- Open-source community for tools and feedback
- Contributors who help improve this guide

---

## 📬 Contact

For questions or suggestions:

- **LinkedIn**: [Navneet Shukla](https://www.linkedin.com/in/navneet-shukla17/)
- **Email**: shuklanavneet2817@gmail.com

---

<div align="center">

**Built with ❤️ by Navneet Shukla**

*Last Updated: 24, December 2025*

</div>
