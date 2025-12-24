import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import load_prompt
import time

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(
    page_title="AI Research Summarizer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Custom CSS to hide default header and improve font
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
    }
    .summary-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #ff4b4b;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Model (Cached to prevent reloading on every interaction)
@st.cache_resource
def get_model():
    return ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')

try:
    model = get_model()
except Exception as e:
    st.error(f"Error loading model. Check your API Key. Details: {e}")
    st.stop()

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.title("⚙️ Configuration")
    
    st.info("Select a paper and customize the explanation style below.")

    paper_options = [
        'Attention Is All You Need — Vaswani et al., 2017',
        'Improved Training of Wasserstein GANs — Gulrajani et al., 2017',
        'BERT: Pre-training of Deep Bidirectional Transformers — Devlin et al., 2018',
        'StyleGAN: A Style-Based Generator Architecture for GANs — Karras et al., 2018',
        'GPT-2: Language Models Are Unsupervised Multitask Learners — Radford et al., 2019',
        'Progressive Growing of GANs — Karras et al., 2019',
        'GPT-3: Language Models Are Few-Shot Learners — Brown et al., 2020',
        'Denoising Diffusion Probabilistic Models (DDPM) — Ho et al., 2020',
        'Stable Diffusion: High-Resolution Image Synthesis — Rombach et al., 2022',
        'PaLM: Scaling Language Modeling with Pathways — Chowdhery et al., 2022',
        'LLaMA: Open and Efficient Foundation Language Models — Touvron et al., 2023',
        'Toolformer: Language Models Can Teach Themselves — Schick et al., 2023',
        'Sora: A Large-Scale Diffusion Transformer — OpenAI, 2024',
        'Gemini: A Multimodal LLM — Google DeepMind, 2024',
        'OpenAI o1: Reasoning Models — OpenAI, 2025',
        'Qwen2.5: Next-Gen Open LLM — Alibaba, 2025',
        'DeepSeek-V3: Highly Optimized Transformer — DeepSeek, 2025',
        'Flux: A Diffusion Transformer for Video — 2025'
    ]
    
    paper_input = st.selectbox(
        label='📄 Select Research Paper', 
        options=paper_options,
        index=0
    )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        style_input = st.selectbox(
            label='🎭 Explanation Style', 
            options=[
                'Beginner-Friendly',
                'Technical',
                'Code-oriented',
                'Mathematical',
                'Explanatory - Non technical'
            ]
        )
    with col2:
        length_input = st.selectbox(
            label='📏 Length', 
            options=[
                'Short (1-2 paras)',
                'Medium (3-5 paras)',
                'Long (Detailed)'
            ]
        )
    
    st.divider()
    
    submit_btn = st.button(label='✨ Summarize Paper', type="primary")

# --- MAIN CONTENT ---

st.title("📚 Research Paper Summarizer")
st.markdown("Generate concise, tailored summaries of landmark AI research papers using **Gemini 2.5**.")

# Check if Prompt Template exists
try:
    template = load_prompt('template.json')
except Exception:
    st.warning("⚠️ `template.json` not found. Using default prompt.")
    from langchain_core.prompts import PromptTemplate
    template = PromptTemplate.from_template(
        "Summarize the paper '{paper_input}' in a {style_input} style. Keep it {length_input}."
    )

if submit_btn:
    if not paper_input:
        st.warning("Please select a paper first.")
    else:
        # Create container for visual grouping
        result_container = st.container()
        
        with result_container:
            # Spinner for UX
            with st.spinner(f"Reading '{paper_input}'..."):
                try:
                    chain = template | model
                    
                    # Create a placeholder for streaming output
                    response_placeholder = st.empty()
                    full_response = ""

                    # Stream the response
                    chunks = chain.stream({
                        'paper_input': paper_input,
                        'style_input': style_input,
                        'length_input': length_input
                    })
                    
                    # Markdown styling wrapper
                    st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                    
                    # Display stream
                    for chunk in chunks:
                        full_response += chunk.content
                        response_placeholder.markdown(full_response + "▌")
                    
                    response_placeholder.markdown(full_response)
                    
                    st.markdown('</div>', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

# Add a footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey;'>Powered by LangChain & Google Gemini</div>", 
    unsafe_allow_html=True
)