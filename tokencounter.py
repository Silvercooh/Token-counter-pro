import streamlit as st
import tiktoken
import json
from typing import Dict, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Try to import SentencePiece for Grok models
try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Token Counter Pro", 
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .token-display {
        font-family: 'Courier New', monospace;
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🧮 Token Counter Pro")
st.markdown("**Analyze token usage across different OpenAI models with advanced features**")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection with categories
    st.subheader("Model Selection")
    model_categories = {
        "🚀 GPT-4 Series (cl100k_base)": [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4-turbo-preview",
            "gpt-4-1106-preview",
            "gpt-4-0125-preview",
            "gpt-4-vision-preview",
            "gpt-4-32k",
            "gpt-4-0613",
            "gpt-4-32k-0613"
        ],
        "⚡ GPT-3.5 Series (cl100k_base)": [
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-3.5-turbo-instruct"
        ],
        "🎯 GPT-3 Davinci (p50k_base)": [
            "text-davinci-003",
            "text-davinci-002",
            "text-davinci-001",
            "davinci",
            "davinci-instruct-beta",
            "davinci-002"
        ],
        "🔤 GPT-3 Other (p50k_base)": [
            "text-curie-001",
            "text-babbage-001",
            "text-ada-001",
            "curie",
            "babbage",
            "ada",
            "curie-instruct-beta",
            "babbage-002",
            "davinci-similarity"
        ],
        "🧠 GPT-2 Series (gpt2)": [
            "gpt2",
            "text-davinci-edit-001",
            "code-davinci-edit-001"
        ],
        "💬 Chat Models (cl100k_base)": [
            "gpt-3.5-turbo-0301",
            "gpt-4-0314",
            "gpt-4-32k-0314"
        ],
        "🔧 Code Models (p50k_base)": [
            "code-davinci-002",
            "code-davinci-001",
            "code-cushman-002",
            "code-cushman-001"
        ],
        "🎨 DALL-E (cl100k_base)": [
            "dall-e-2",
            "dall-e-3"
        ],
        "📝 Embedding Models": [
            "text-embedding-ada-002",
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-similarity-davinci-001",
            "text-similarity-curie-001",
            "text-similarity-babbage-001",
            "text-similarity-ada-001"
        ],
        "🔍 Search Models": [
            "text-search-davinci-doc-001",
            "text-search-curie-doc-001",
            "text-search-babbage-doc-001",
            "text-search-ada-doc-001"
        ],
        "🧪 Custom Encodings": [
            "cl100k_base",
            "p50k_base", 
            "p50k_edit",
            "r50k_base",
            "gpt2"
        ],
        "🤖 xAI Grok (SentencePiece)": [
            "grok-1",
            "grok-1.5",
            "grok-2",
            "grok-beta",
            "grok-vision-beta"
        ]
    }
    
    selected_models = []
    for category_idx, (category, models) in enumerate(model_categories.items()):
        st.write(f"**{category}**")
        for model_idx, model in enumerate(models):
            unique_key = f"model_{category_idx}_{model_idx}_{model}"
            if st.checkbox(model, key=unique_key, value=(model == "gpt-3.5-turbo")):
                selected_models.append(model)
    
    if not selected_models:
        selected_models = ["gpt-3.5-turbo"]  # Default fallback
    
    st.divider()
    
    # Quick actions
    st.subheader("🎯 Quick Actions")
    quick_col1, quick_col2, quick_col3 = st.columns(3)
    
    with quick_col1:
        if st.button("🔄 Reset All", use_container_width=True):
            st.rerun()
    
    with quick_col2:
        if st.button("📋 Popular Models", use_container_width=True):
            st.session_state.quick_select = "popular"
    
    with quick_col3:
        if st.button("🧪 All Encodings", use_container_width=True):
            st.session_state.quick_select = "encodings"
    
    # Handle quick selections
    if hasattr(st.session_state, 'quick_select'):
        if st.session_state.quick_select == "popular":
            # Auto-select popular models
            popular_models = ["gpt-4", "gpt-3.5-turbo", "text-davinci-003", "gpt2"]
            for category_idx, (category, models) in enumerate(model_categories.items()):
                for model_idx, model in enumerate(models):
                    if model in popular_models:
                        unique_key = f"model_{category_idx}_{model_idx}_{model}"
                        st.session_state[unique_key] = True
            st.session_state.quick_select = None
        elif st.session_state.quick_select == "encodings":
            # Auto-select encoding representatives
            encoding_models = ["cl100k_base", "p50k_base", "p50k_edit", "r50k_base", "gpt2"]
            for category_idx, (category, models) in enumerate(model_categories.items()):
                for model_idx, model in enumerate(models):
                    if model in encoding_models:
                        unique_key = f"model_{category_idx}_{model_idx}_{model}"
                        st.session_state[unique_key] = True
            st.session_state.quick_select = None
    
    # Display options
    st.subheader("Display Options")
    show_tokens = st.checkbox("Show individual tokens", value=False)
    show_stats = st.checkbox("Show detailed statistics", value=True)
    show_comparison = st.checkbox("Show model comparison", value=len(selected_models) > 1)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Text Input")
    
    # Input options
    input_method = st.radio(
        "Input method:",
        ["Text Area", "File Upload", "Sample Texts"],
        horizontal=True
    )
    
    text = ""
    
    if input_method == "Text Area":
        text = st.text_area(
            "Enter your text below:",
            height=300,
            placeholder="Type or paste your text here..."
        )
    
    elif input_method == "File Upload":
        uploaded_file = st.file_uploader(
            "Choose a text file",
            type=['txt', 'md', 'py', 'js', 'json']
        )
        if uploaded_file is not None:
            text = str(uploaded_file.read(), "utf-8")
            st.text_area("File content preview:", value=text[:500] + "..." if len(text) > 500 else text, height=150, disabled=True)
    
    else:  # Sample Texts
        samples = {
            "Short message": "Hello, how are you doing today?",
            "Code snippet": """def hello_world():
    print("Hello, World!")
    return "Success" """,
            "Long paragraph": """The quick brown fox jumps over the lazy dog. This pangram contains every letter of the English alphabet at least once, making it useful for testing fonts and keyboards. It has been used since the late 19th century and remains popular in typography and computer testing.""",
            "JSON example": json.dumps({"name": "John", "age": 30, "city": "New York", "hobbies": ["reading", "swimming", "coding"]}, indent=2),
            "Multilingual text": "Hello! 你好! Bonjour! Hola! こんにちは! Привет! مرحبا! 안녕하세요!",
            "Special characters": "Here are some special chars: @#$%^&*()_+-=[]{}|;':\",./<>?`~",
            "Emoji test": "I love coding! 💻🚀 Let's build something amazing! 🎉✨🔥💡",
            "Mathematical notation": "E=mc², ∫f(x)dx, √(a²+b²), Σ(i=1 to n), ∂f/∂x",
            "Markdown example": """# Heading 1
## Heading 2
**Bold text** and *italic text*
- List item 1
- List item 2
```python
print("Hello World")
```""",
            "Prompt engineering": """You are an AI assistant. Please follow these instructions:
1. Be helpful and accurate
2. Explain complex topics clearly  
3. Ask clarifying questions when needed
4. Provide examples when appropriate"""
        }
        
        selected_sample = st.selectbox("Choose a sample text:", list(samples.keys()))
        text = samples[selected_sample]
        st.text_area("Sample text:", value=text, height=150, disabled=True)

with col2:
    st.subheader("📊 Quick Stats")
    
    if text:
        # Character and word counts
        char_count = len(text)
        word_count = len(text.split())
        line_count = len(text.split('\n'))
        
        st.metric("Characters", f"{char_count:,}")
        st.metric("Words", f"{word_count:,}")
        st.metric("Lines", line_count)
    else:
        st.info("Enter text to see statistics")

# Token analysis
if text and st.button("🔍 Analyze Tokens", type="primary", use_container_width=True):
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = {}
    
    # Encoding mappings for better fallback handling
    encoding_map = {
        # Direct encoding names
        "cl100k_base": "cl100k_base",
        "p50k_base": "p50k_base", 
        "p50k_edit": "p50k_edit",
        "r50k_base": "r50k_base",
        "gpt2": "gpt2",
        
        # GPT-4 models
        "gpt-4": "cl100k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-4-32k": "cl100k_base",
        
        # GPT-3.5 models  
        "gpt-3.5-turbo": "cl100k_base",
        "gpt-3.5-turbo-16k": "cl100k_base",
        "gpt-3.5-turbo-instruct": "cl100k_base",
        
        # GPT-3 Davinci models
        "text-davinci-003": "p50k_base",
        "text-davinci-002": "p50k_base",
        "text-davinci-001": "p50k_base",
        "davinci": "p50k_base",
        
        # Other GPT-3 models
        "text-curie-001": "p50k_base",
        "text-babbage-001": "p50k_base", 
        "text-ada-001": "p50k_base",
        "curie": "p50k_base",
        "babbage": "p50k_base",
        "ada": "p50k_base",
        
        # Code models
        "code-davinci-002": "p50k_base",
        "code-davinci-001": "p50k_base",
        "code-cushman-002": "p50k_base",
        "code-cushman-001": "p50k_base",
        
        # Edit models
        "text-davinci-edit-001": "p50k_edit",
        "code-davinci-edit-001": "p50k_edit",
        
        # Embedding models
        "text-embedding-ada-002": "cl100k_base",
        "text-embedding-3-small": "cl100k_base",
        "text-embedding-3-large": "cl100k_base",
        
        # DALL-E
        "dall-e-2": "cl100k_base",
        "dall-e-3": "cl100k_base",
        
        # Grok models (SentencePiece - handled separately)
        "grok-1": "sentencepiece",
        "grok-1.5": "sentencepiece", 
        "grok-2": "sentencepiece",
        "grok-beta": "sentencepiece",
        "grok-vision-beta": "sentencepiece"
    }
    
    for i, model in enumerate(selected_models):
        status_text.text(f'Analyzing with {model}...')
        progress_bar.progress((i + 1) / len(selected_models))
        
        # Smart encoding detection
        encoding = None
        encoding_name = "unknown"
        is_sentencepiece = False
        
        try:
            # First try tiktoken's built-in model mapping
            encoding = tiktoken.encoding_for_model(model)
            encoding_name = model
        except KeyError:
            # Fall back to our custom mapping
            if model in encoding_map:
                try:
                    encoding_name = encoding_map[model]
                    
                    if encoding_name == "sentencepiece":
                        is_sentencepiece = True
                        if not SENTENCEPIECE_AVAILABLE:
                            st.warning(f"⚠️ {model} uses SentencePiece tokenization. Install sentencepiece: `pip install sentencepiece`")
                            continue
                        # For now, we'll use a simple character-based approximation for Grok
                        # In a real implementation, you'd load the actual Grok tokenizer model
                        encoding_name = "grok_approximation"
                    else:
                        encoding = tiktoken.get_encoding(encoding_name)
                except:
                    pass
            
            # Final fallback logic
            if encoding is None and not is_sentencepiece:
                if any(x in model.lower() for x in ["gpt-4", "gpt-3.5", "dall-e", "embedding-3"]):
                    encoding_name = "cl100k_base"
                elif any(x in model.lower() for x in ["davinci", "curie", "babbage", "ada", "code-", "cushman"]):
                    encoding_name = "p50k_base"
                elif "edit" in model.lower():
                    encoding_name = "p50k_edit"
                elif "grok" in model.lower():
                    is_sentencepiece = True
                    encoding_name = "grok_approximation"
                elif "gpt2" in model.lower():
                    encoding_name = "gpt2"
                else:
                    encoding_name = "cl100k_base"  # Most common modern encoding
                
                if not is_sentencepiece:
                    try:
                        encoding = tiktoken.get_encoding(encoding_name)
                    except:
                        st.warning(f"Could not find encoding for {model}, using cl100k_base")
                        encoding = tiktoken.get_encoding("cl100k_base")
                        encoding_name = "cl100k_base"
        
        # Handle SentencePiece models (Grok)
        if is_sentencepiece:
            if SENTENCEPIECE_AVAILABLE:
                # Simple approximation for Grok tokenization
                # In practice, you'd load the actual Grok tokenizer model from xAI
                # For demonstration, we'll use a word-based approximation
                words = text.split()
                # Grok typically has ~1.3-1.5 tokens per word for English text
                estimated_tokens = int(len(words) * 1.4)
                tokens = list(range(estimated_tokens))  # Dummy token IDs
                unique_tokens = tokens  # All unique for approximation
                token_lengths = [len(word) for word in words] + [1] * (estimated_tokens - len(words))
            else:
                continue
        else:
            tokens = encoding.encode(text)
            token_lengths = [len(encoding.decode([token])) for token in tokens]
            unique_tokens = list(set(tokens))
        
        results[model] = {
            'tokens': tokens,
            'encoding': encoding if not is_sentencepiece else None,
            'encoding_name': encoding_name,
            'is_sentencepiece': is_sentencepiece,
            'count': len(tokens),
            'unique_count': len(unique_tokens),
            'avg_length': sum(token_lengths) / len(token_lengths) if token_lengths else 0,
            'max_length': max(token_lengths) if token_lengths else 0,
            'min_length': min(token_lengths) if token_lengths else 0
        }
    
    progress_bar.empty()
    status_text.empty()
    
    # Display results
    st.success("✅ Analysis Complete!")
    
    # Token count comparison
    if show_comparison and len(selected_models) > 1:
        st.subheader("📈 Model Comparison")
        
        # Create comparison chart
        fig = go.Figure()
        
        models = list(results.keys())
        token_counts = [results[model]['count'] for model in models]
        unique_counts = [results[model]['unique_count'] for model in models]
        
        fig.add_trace(go.Bar(
            name='Total Tokens',
            x=models,
            y=token_counts,
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Unique Tokens',
            x=models,
            y=unique_counts,
            marker_color='lightcoral'
        ))
        
        fig.update_layout(
            title='Token Count Comparison Across Models',
            xaxis_title='Models',
            yaxis_title='Token Count',
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Comparison table
        comparison_data = []
        for model in models:
            r = results[model]
            comparison_data.append({
                'Model': model,
                'Encoding': r['encoding_name'],
                'Total Tokens': f"{r['count']:,}",
                'Unique Tokens': f"{r['unique_count']:,}",
                'Avg Token Length': f"{r['avg_length']:.2f}",
                'Efficiency*': f"{(len(text) / r['count']):.2f}"
            })
        
        st.dataframe(comparison_data, use_container_width=True)
        st.caption("*Efficiency = Characters per token (higher is more efficient)")
    
    # Detailed results for each model
    for model in selected_models:
        result = results[model]
        
        st.subheader(f"🔍 {model} Analysis")
        if result.get('is_sentencepiece'):
            st.caption(f"Using tokenizer: **{result['encoding_name']}** (SentencePiece - Approximated)")
            st.info("📝 **Note:** Grok models use SentencePiece tokenization. Token counts shown are estimated approximations.")
        else:
            st.caption(f"Using encoding: **{result['encoding_name']}**")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Tokens",
                f"{result['count']:,}",
                delta=f"vs {char_count} chars" if len(selected_models) == 1 else None
            )
        
        with col2:
            st.metric("Unique Tokens", f"{result['unique_count']:,}")
        
        with col3:
            compression_ratio = (char_count / result['count']) if result['count'] > 0 else 0
            st.metric("Compression Ratio", f"{compression_ratio:.2f}:1")
        
        with col4:
            st.metric("Avg Token Length", f"{result['avg_length']:.2f}")
        
        # Detailed statistics
        if show_stats:
            with st.expander(f"📊 Detailed Statistics - {model}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Token Length Statistics:**")
                    st.write(f"• Minimum: {result['min_length']} characters")
                    st.write(f"• Maximum: {result['max_length']} characters")
                    st.write(f"• Average: {result['avg_length']:.2f} characters")
                
                with col2:
                    st.write("**Text Statistics:**")
                    st.write(f"• Characters per token: {compression_ratio:.2f}")
                    st.write(f"• Words per token: {(word_count / result['count']):.2f}")
                    st.write(f"• Token efficiency: {((result['count'] / char_count) * 100):.1f}%")
        
        # Show individual tokens
        if show_tokens and not result.get('is_sentencepiece'):
            with st.expander(f"🔤 Individual Tokens - {model} ({result['count']} tokens)"):
                if result['count'] > 1000:
                    st.warning(f"⚠️ Large number of tokens ({result['count']}). Showing first 100 for performance.")
                    display_tokens = result['tokens'][:100]
                else:
                    display_tokens = result['tokens']
                
                cols = st.columns(3)
                for i, token in enumerate(display_tokens):
                    decoded = result['encoding'].decode([token])
                    col_idx = i % 3
                    
                    with cols[col_idx]:
                        # Escape special characters for display
                        display_decoded = repr(decoded)[1:-1]  # Remove outer quotes
                        st.markdown(f"`{i+1:3d}` **{token:5d}** → `{display_decoded}`")
                
                if result['count'] > len(display_tokens):
                    st.info(f"... and {result['count'] - len(display_tokens)} more tokens")
        elif show_tokens and result.get('is_sentencepiece'):
            with st.expander(f"🔤 Token Information - {model}"):
                st.info("Individual token display not available for SentencePiece models (Grok). Only token count estimation is provided.")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>
        Token counts are approximate and may vary slightly from actual API usage. 
        This tool uses tiktoken for OpenAI models and approximation for Grok models (SentencePiece).
        <br>
        <strong>Grok Note:</strong> Grok models use SentencePiece tokenization. For accurate counts, use the official xAI tokenizer.
    </small>
</div>
""", unsafe_allow_html=True)