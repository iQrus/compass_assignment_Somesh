import streamlit as st
import pandas as pd
import numpy as np
import time
import logging
import os
from typing import Dict, List, Any, Tuple, Optional
import plotly.express as px
import plotly.graph_objects as go
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Import our modules
from product_matcher import (
    load_data, 
    load_validation_data, 
    create_vector_store, 
    match_products_batch,
    evaluate_validation_set
)
from dashboard import ProductMatchingDashboard

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ProductMatcherApp")

# Use Streamlit's caching for data and resource-intensive operations
@st.cache_data
def load_cached_data():
    """Load and cache external and internal data.
    
    This function will only reload data when the function itself changes
    or when it's explicitly cleared from the cache.
    """
    logger.info("Loading and caching data...")
    external_df, internal_df = load_data()
    validation_df = load_validation_data()
    return external_df, internal_df, validation_df

@st.cache_resource
def create_cached_vector_store(_internal_df):
    """Create and cache vector store using text-embedding-3-small model.
    
    Args:
        _internal_df: DataFrame containing internal products
        
    Returns:
        FAISS vector store
    """
    model_name = "text-embedding-3-small"
    logger.info(f"Creating and caching vector store with model {model_name}...")
    return create_vector_store(_internal_df, "LONG_NAME", model_name)

def initialize_data():
    """Initialize data and embeddings using caching.
    
    Returns:
        Tuple of (external_df, internal_df, validation_df, vector_store)
    """
    try:
        # Load data (this will be cached)
        external_df, internal_df, validation_df = load_cached_data()
        
        # Create vector store (this will be cached based on internal_df)
        vector_store = create_cached_vector_store(internal_df)
        
        logger.info("Data initialization complete")
        return external_df, internal_df, validation_df, vector_store
    except Exception as e:
        logger.error(f"Data initialization error: {str(e)}", exc_info=True)
        raise e

def main():
    """Main application entry point."""
    # Page config
    st.set_page_config(
        page_title="AI Product Matcher",
        page_icon="🏪",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🏪 AI Product Matcher")
    st.markdown("""
    This application uses AI to automatically match external products with internal products.
    It combines vector embeddings for similarity search with LLM-based matching for high accuracy.
    """)
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = st.text_input("Enter your OpenAI API key:", type="password")
        if not api_key:
            st.warning("Please enter your OpenAI API key to continue.")
            st.stop()
        else:
            os.environ["OPENAI_API_KEY"] = api_key
    
    # Sidebar
    st.sidebar.header("Model Parameters")
    
    # LLM model selection
    model_name = st.sidebar.selectbox(
        "LLM Model",
        ["gpt-3.5-turbo", "gpt-4"],
        index=1,
        help="Select the OpenAI model to use for matching. GPT-4 is more accurate but slower and more expensive."
    )
    
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Controls randomness in the model's output. Lower values mean more deterministic responses."
    )
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=50.0,
        max_value=100.0,
        value=70.0,
        step=5.0,
        help="Minimum confidence score required to consider a match valid. Higher values mean stricter matching."
    )
    
    top_k = st.sidebar.slider(
        "Top K Similar Products",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="Number of candidate products to consider for each external product."
    )
    
    batch_size = st.sidebar.slider(
        "Number of Products per Batch",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="Number of external products to consider for each batch."
    )
    
    # Initialize data
    try:
        external_df, internal_df, validation_df, vector_store = initialize_data()
    except Exception as e:
        st.error(f"Error initializing data: {str(e)}")
        logger.error(f"Data initialization error: {str(e)}", exc_info=True)
        st.stop()
    
    # Add dataset statistics to sidebar
    st.sidebar.header("Dataset Statistics")
    st.sidebar.write(f"External Products: {len(external_df)}")
    st.sidebar.write(f"Internal Products: {len(internal_df)}")
    st.sidebar.write(f"Validation Entries: {len(validation_df)}")
    
    # Add cache management
    st.sidebar.header("Cache Management")
    cache_cols = st.sidebar.columns(2)
    if cache_cols[0].button("Clear Data Cache"):
        # Clear only the data cache
        load_cached_data.clear()
        st.sidebar.success("Data cache cleared!")
        time.sleep(1)
        st.experimental_rerun()
        
    if cache_cols[1].button("Clear All Caches"):
        # Clear all caches
        load_cached_data.clear()
        create_cached_vector_store.clear()
        st.sidebar.success("All caches cleared!")
        time.sleep(1)
        st.experimental_rerun()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["Match Products", "Validation", "Data Explorer"])
    
    # Tab 1: Product Matching
    with tab1:
        st.header("Match External and Internal Products")
        
        # Display sample data
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("External Products Sample")
            st.dataframe(external_df.head(5), hide_index=True)
        
        with col2:
            st.subheader("Internal Products Sample")
            st.dataframe(internal_df.head(5), hide_index=True)
        
        # Run matching button
        if st.button("Run Product Matching", type="primary"):
            with st.spinner("Matching products..."):
                try:
                    # Run the matching algorithm
                    results_df = match_products_batch(
                        external_df,
                        internal_df,
                        vector_store,
                        model_name=model_name,
                        temperature=temperature,
                        confidence_threshold=confidence_threshold,
                        top_k=top_k,
                        batch_size=batch_size
                    )
                    
                    # Evaluate against validation set
                    validation_metrics = evaluate_validation_set(results_df, validation_df)
                    
                    # Store results in session state
                    st.session_state["results_df"] = results_df
                    st.session_state["validation_metrics"] = validation_metrics
                    
                    # Display dashboard
                    dashboard = ProductMatchingDashboard(results_df, validation_metrics)
                    dashboard.render_overview()
                    dashboard.render_match_details()
                    dashboard.render_example_match()
                    
                except Exception as e:
                    st.error(f"Error during matching: {str(e)}")
                    logger.error(f"Matching error: {str(e)}", exc_info=True)
    
    # Tab 2: Validation
    with tab2:
        st.header("Validation Results")
        
        if "results_df" in st.session_state and "validation_metrics" in st.session_state:
            # Display validation results
            dashboard = ProductMatchingDashboard(
                st.session_state["results_df"],
                st.session_state["validation_metrics"]
            )
            dashboard.render_validation_results(validation_df)
        else:
            st.info("Run the matching process first to see validation results")
    
    # Tab 3: Data Explorer
    with tab3:
        st.header("Data Explorer")
        
        # Search functionality
        st.subheader("Search Products")
        search_term = st.text_input("Search term:")
        
        if search_term:
            # Search in external products
            external_matches = external_df[external_df["PRODUCT_NAME"].str.contains(search_term, case=False)]
            
            # Search in internal products
            internal_matches = internal_df[internal_df["LONG_NAME"].str.contains(search_term, case=False)]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"External matches ({len(external_matches)}):")
                st.dataframe(external_matches, hide_index=True)
            
            with col2:
                st.write(f"Internal matches ({len(internal_matches)}):")
                st.dataframe(internal_matches.head(100), hide_index=True)
                if len(internal_matches) > 100:
                    st.info(f"Showing 100 out of {len(internal_matches)} matches")
        
        # Data distributions
        st.subheader("Data Distributions & Analytics")

        tab_external, tab_internal, tab_comparative = st.tabs(["External Products", "Internal Products", "Comparative Analysis"])

        with tab_external:
            st.write("### External Product Analytics")
            
            # Create two columns for analytics
            col1, col2 = st.columns(2)
            
            with col1:
                try:
                    st.write("#### Word Cloud of External Products")
                    
                    # Create a string of all product names
                    text = ' '.join(external_df["PRODUCT_NAME"].str.lower())
                    
                    # Remove common stopwords
                    stopwords = ['the', 'and', 'a', 'an', 'in', 'on', 'at', 'with', 'by', 'for', 'to', 'of', 'is']
                    for word in stopwords:
                        text = text.replace(f" {word} ", " ")
                    
                    # Generate the word cloud
                    wordcloud = WordCloud(
                        width=800, 
                        height=400, 
                        background_color='white', 
                        max_words=100,
                        colormap='Blues',
                        contour_width=1,
                        contour_color='steelblue'
                    ).generate(text)
                    
                    # Display the word cloud using Matplotlib
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig)
                except ImportError:
                    st.warning("WordCloud library not installed. Install with `pip install wordcloud`.")
                    
                    # Fallback to bar chart of common words
                    st.write("#### Most Common Words in External Products")
                    all_words = ' '.join(external_df["PRODUCT_NAME"].str.lower()).split()
                    word_counts = pd.Series(all_words).value_counts().head(20)
                    # Filter out stopwords
                    word_counts = word_counts[~word_counts.index.isin(stopwords)]
                    st.bar_chart(word_counts)
            
            with col2:
                # Size distribution analysis
                st.write("#### Size Information Analysis")
                
                # Extract size information using regex
                import re
                
                def extract_size(name):
                    # Look for patterns like "12 OZ", "16oz", "1.5 lb", "(20oz)" etc.
                    size_pattern = r'(\d+(\.\d+)?)\s*(oz|lb|g|ml|l|fl\s*oz|gal|kg|ct|pack|\d+\s*pk|count)'
                    size_match = re.search(size_pattern, name, re.IGNORECASE)
                    if size_match:
                        return size_match.group(0).strip()
                    return None
                
                # Apply the extraction to all product names
                external_df['size_info'] = external_df["PRODUCT_NAME"].apply(extract_size)
                
                # Count products with size information
                has_size = external_df['size_info'].notna().sum()
                no_size = len(external_df) - has_size
                
                # Create pie chart for size presence
                labels = ['With Size Info', 'Without Size Info']
                values = [has_size, no_size]
                
                fig = px.pie(
                    values=values,
                    names=labels,
                    title="Products with Size Information",
                    color_discrete_sequence=px.colors.sequential.Blues
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Most common sizes
                if has_size > 0:
                    st.write("#### Most Common Product Sizes")
                    size_counts = external_df['size_info'].value_counts().head(10)
                    fig = px.bar(
                        x=size_counts.index,
                        y=size_counts.values,
                        labels={"x": "Size", "y": "Count"},
                        title="Top 10 Product Sizes",
                        color=size_counts.values,
                        color_continuous_scale="Blues"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Third section for advanced analytics
            st.write("#### Product Categorization Analysis")
            
            # Create word-based categorization
            category_keywords = {
                'Beverages': ['tea', 'coffee', 'water', 'juice', 'milk', 'cola', 'drink', 'energy', 'soda', 'coke', 'pepsi'],
                'Snacks': ['chip', 'cookie', 'chocolate', 'candy', 'snack', 'bar', 'cracker', 'popcorn'],
                'Bakery': ['bread', 'bagel', 'cake', 'muffin', 'donut', 'danish', 'bun', 'roll', 'pastry'],
                'Dairy': ['milk', 'cheese', 'yogurt', 'cream', 'butter', 'dairy'],
                'Meat': ['meat', 'beef', 'chicken', 'pork', 'sausage', 'bacon', 'ham', 'turkey'],
                'Produce': ['fruit', 'vegetable', 'apple', 'banana', 'orange', 'fresh', 'salad', 'produce'],
                'Household': ['paper', 'cleaner', 'detergent', 'soap', 'tissue', 'household'],
                'Health & Beauty': ['shampoo', 'lotion', 'soap', 'toothpaste', 'deodorant']
            }
            
            def categorize_product(name):
                name_lower = name.lower()
                for category, keywords in category_keywords.items():
                    if any(keyword in name_lower for keyword in keywords):
                        return category
                return 'Other'
            
            external_df['category'] = external_df["PRODUCT_NAME"].apply(categorize_product)
            
            # Display category distribution
            category_counts = external_df['category'].value_counts()
            
            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Product Categories (Keyword-Based)",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab_internal:
            st.write("### Internal Product Analytics")
            
            # Sample for performance
            sample_df = internal_df.sample(min(1000, len(internal_df)), random_state=42)
            
            # Create two columns for analytics
            col1, col2 = st.columns(2)
            
            with col1:
                try:
                    st.write("#### Word Cloud of Internal Products (Sample)")
                    
                    # Create a string of all product names
                    text = ' '.join(sample_df["LONG_NAME"].str.lower())
                    
                    # Remove common stopwords
                    stopwords = ['the', 'and', 'a', 'an', 'in', 'on', 'at', 'with', 'by', 'for', 'to', 'of', 'is']
                    for word in stopwords:
                        text = text.replace(f" {word} ", " ")
                    
                    # Generate the word cloud
                    wordcloud = WordCloud(
                        width=800, 
                        height=400, 
                        background_color='white', 
                        max_words=100,
                        colormap='Blues',
                        contour_width=1,
                        contour_color='steelblue'
                    ).generate(text)
                    
                    # Display the word cloud using Matplotlib
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig)
                except ImportError:
                    st.warning("WordCloud library not installed. Install with `pip install wordcloud`.")
                    
                    # Fallback to bar chart of common words
                    st.write("#### Most Common Words in Internal Products")
                    all_words = ' '.join(sample_df["LONG_NAME"].str.lower()).split()
                    word_counts = pd.Series(all_words).value_counts().head(20)
                    # Filter out stopwords
                    word_counts = word_counts[~word_counts.index.isin(stopwords)]
                    st.bar_chart(word_counts)
            
            with col2:
                # Size distribution analysis for internal products
                st.write("#### Size Information Analysis")
                
                # Extract size information using regex
                def extract_size(name):
                    # Look for patterns like "12 OZ", "16oz", "1.5 lb", "(20oz)" etc.
                    size_pattern = r'(\d+(\.\d+)?)\s*(oz|lb|g|ml|l|fl\s*oz|gal|kg|ct|pack|\d+\s*pk|count)'
                    size_match = re.search(size_pattern, name, re.IGNORECASE)
                    if size_match:
                        return size_match.group(0).strip()
                    
                    # Also look for patterns like "(20oz)" which are common in internal products
                    paren_pattern = r'\((\d+(\.\d+)?)(oz|lb|g|ml|l)\)'
                    paren_match = re.search(paren_pattern, name, re.IGNORECASE)
                    if paren_match:
                        return paren_match.group(0).strip()
                    
                    return None
                
                # Apply the extraction to sample of internal products
                sample_df['size_info'] = sample_df["LONG_NAME"].apply(extract_size)
                
                # Count products with size information
                has_size = sample_df['size_info'].notna().sum()
                no_size = len(sample_df) - has_size
                
                # Create pie chart for size presence
                labels = ['With Size Info', 'Without Size Info']
                values = [has_size, no_size]
                
                fig = px.pie(
                    values=values,
                    names=labels,
                    title="Products with Size Information (Sample)",
                    color_discrete_sequence=px.colors.sequential.Blues
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Most common sizes
                if has_size > 0:
                    st.write("#### Most Common Product Sizes")
                    size_counts = sample_df['size_info'].value_counts().head(10)
                    fig = px.bar(
                        x=size_counts.index,
                        y=size_counts.values,
                        labels={"x": "Size", "y": "Count"},
                        title="Top 10 Product Sizes (Sample)",
                        color=size_counts.values,
                        color_continuous_scale="Blues"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Format analysis
            st.write("#### Format Analysis of Internal Products")
            
            # Check for parentheses in size format
            has_parentheses = sample_df["LONG_NAME"].str.contains(r'\(\d+(\.\d+)?(oz|lb|g|ml|l)\)', case=False).sum()
            has_space_unit = sample_df["LONG_NAME"].str.contains(r'\d+(\.\d+)?\s+(oz|lb|g|ml|l)', case=False).sum()
            other_size_format = has_size - has_parentheses - has_space_unit
            
            # Create pie chart for size format
            labels = ['Parentheses Format (20oz)', 'Space Format (20 oz)', 'Other Format']
            values = [has_parentheses, has_space_unit, other_size_format]
            
            fig = px.pie(
                values=values,
                names=labels,
                title="Size Format Distribution (Sample)",
                color_discrete_sequence=px.colors.sequential.Purples
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab_comparative:
            st.write("### Comparative Analysis")
            
            # Create a size format comparison
            st.write("#### Size Format Comparison")
            
            # Count products with different size formats in external dataset
            external_parentheses = external_df["PRODUCT_NAME"].str.contains(r'\(\d+(\.\d+)?(oz|lb|g|ml|l)\)', case=False).sum()
            external_space_unit = external_df["PRODUCT_NAME"].str.contains(r'\d+(\.\d+)?\s+(oz|lb|g|ml|l)', case=False).sum()
            
            # Count for internal (sample)
            internal_parentheses = sample_df["LONG_NAME"].str.contains(r'\(\d+(\.\d+)?(oz|lb|g|ml|l)\)', case=False).sum()
            internal_space_unit = sample_df["LONG_NAME"].str.contains(r'\d+(\.\d+)?\s+(oz|lb|g|ml|l)', case=False).sum()
            
            # Create a comparison dataframe
            format_comparison = pd.DataFrame({
                'Format': ['Parentheses Format (20oz)', 'Space Format (20 oz)'],
                'External': [external_parentheses, external_space_unit],
                'Internal': [internal_parentheses, internal_space_unit]
            })
            
            # Normalize to percentages
            format_comparison['External %'] = format_comparison['External'] / len(external_df) * 100
            format_comparison['Internal %'] = format_comparison['Internal'] / len(sample_df) * 100
            
            # Create a grouped bar chart
            fig = px.bar(
                format_comparison,
                x='Format',
                y=['External %', 'Internal %'],
                title="Size Format Comparison (Percentage)",
                barmode='group',
                color_discrete_sequence=['#1f77b4', '#ff7f0e']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Word presence comparison
            st.write("#### Common Word Presence Comparison")
            
            # Common words to check in both datasets
            common_words = ['chocolate', 'water', 'energy', 'coffee', 'tea', 'milk', 'soda', 'candy', 'chip', 'fruit']
            
            # Create comparison data
            word_comparison = []
            
            for word in common_words:
                external_percent = external_df["PRODUCT_NAME"].str.contains(word, case=False).mean() * 100
                internal_percent = sample_df["LONG_NAME"].str.contains(word, case=False).mean() * 100
                word_comparison.append({
                    'Word': word,
                    'External %': external_percent,
                    'Internal %': internal_percent
                })
            
            word_comparison_df = pd.DataFrame(word_comparison)
            
            # Create a grouped bar chart
            fig = px.bar(
                word_comparison_df,
                x='Word',
                y=['External %', 'Internal %'],
                title="Word Presence Comparison",
                barmode='group',
                color_discrete_sequence=['#1f77b4', '#ff7f0e']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Case analysis
            st.write("#### Case Format Analysis")
            
            # Check case format in external products
            external_uppercase = external_df["PRODUCT_NAME"].str.isupper().mean() * 100
            external_titlecase = (external_df["PRODUCT_NAME"].str.istitle() | 
                                (external_df["PRODUCT_NAME"].str.split().apply(lambda x: all(word.istitle() for word in x if word.isalpha())))).mean() * 100
            external_other = 100 - external_uppercase - external_titlecase
            
            # Check case format in internal products
            internal_uppercase = sample_df["LONG_NAME"].str.isupper().mean() * 100
            internal_titlecase = (sample_df["LONG_NAME"].str.istitle() | 
                                (sample_df["LONG_NAME"].str.split().apply(lambda x: all(word.istitle() for word in x if word.isalpha())))).mean() * 100
            internal_other = 100 - internal_uppercase - internal_titlecase
            
            # Create data for case comparison
            case_comparison = pd.DataFrame({
                'Case Format': ['UPPERCASE', 'Title Case', 'Other'],
                'External %': [external_uppercase, external_titlecase, external_other],
                'Internal %': [internal_uppercase, internal_titlecase, internal_other]
            })
            
            # Create a grouped bar chart
            fig = px.bar(
                case_comparison,
                x='Case Format',
                y=['External %', 'Internal %'],
                title="Case Format Comparison",
                barmode='group',
                color_discrete_sequence=['#1f77b4', '#ff7f0e']
            )
            st.plotly_chart(fig, use_container_width=True)

# Run the app
if __name__ == "__main__":
    main()