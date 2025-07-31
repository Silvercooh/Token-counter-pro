import streamlit as st
import json
import re
from typing import Dict, List

# Simple approximation functions (no external dependencies)
def simple_gpt_tokenize(text: str) -> List[str]:
    """Simple GPT-like tokenization approximation"""
    # Split on whitespace and punctuation, similar to GPT behavior
    tokens = re.findall(r'\S+|\s+', text)
    result = []
    for token in tokens:
        if len(token) > 4:  # Split long words
            result.extend([token[i:i+4] for i in range(0, len(token), 3)])
        else:
            result.append(token)
    return result

def estimate_tokens(text: str, model_type: str) -> int:
    """Estimate token count based on model type"""
    char_count = len(text)
    word_count = len(text.split())
    
    # Different models have different compression ratios
    if "gpt-4" in model_type or "gpt-3.5" in model_type:
        return int(word_count * 1.3)  # ~1.3 tokens per word
    elif "davinci" in model_type or "curie" in model_type:
        return int(word_count * 1.4)  # ~1.4 tokens per word  
    elif "grok" in model_type:
        return int(word_count * 1.2)  # ~1.2 tokens per word
    else:
        return int(word_count * 1.3)  # Default

# Page configuration
st.set_page_config(
    page_title="Token Counter (Simple)", 
    page_icon="🧮",
    layout="wide"
)

# Title
st.title("🧮 Token Counter (Deployment-Safe Version)")
st.markdown("**Estimate token usage without external dependencies**")

# Model selection
models = [
    "gpt-4",
    "gpt-3.5-turbo", 
    "text-davinci-003",
    "grok-1",
    "claude-3"
]

selected_models = st.multiselect(
    "Select models to analyze:",
    models,
    default=["gpt-3.5-turbo"]
)

# Text input
text = st.text_area(
    "Enter your text:",
    height=200,
    placeholder="Type your text here..."
)

# Quick stats
if text:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Characters", len(text))
    with col2:
        st.metric("Words", len(text.split()))
    with col3:
        st.metric("Lines", len(text.split('\n')))

# Analysis
if text and selected_models and st.button("🔍 Estimate Tokens"):
    st.success("✅ Analysis Complete!")
    
    results = {}
    for model in selected_models:
        token_count = estimate_tokens(text, model)
        results[model] = {
            'count': token_count,
            'efficiency': len(text) / token_count if token_count > 0 else 0
        }
    
    # Display results
    for model in selected_models:
        result = results[model]
        
        st.subheader(f"📊 {model}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Estimated Tokens", f"{result['count']:,}")
        with col2:
            st.metric("Chars per Token", f"{result['efficiency']:.2f}")
        with col3:
            cost_per_1k = 0.002 if "gpt-4" in model else 0.001
            estimated_cost = (result['count'] / 1000) * cost_per_1k
            st.metric("Est. Cost (USD)", f"${estimated_cost:.4f}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>
        ⚠️ This is a simplified version with estimated token counts.<br>
        For accurate tokenization, use the full version with tiktoken installed.
    </small>
</div>
""", unsafe_allow_html=True)