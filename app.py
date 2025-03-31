import streamlit as st
import pandas as pd
import numpy as np
import time
import logging
import os
from typing import Dict, List, Any, Tuple, Optional

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
    level=logging.INFO,
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
        st.subheader("Data Distributions")
        
        tab_external, tab_internal = st.tabs(["External Products", "Internal Products"])
        
        with tab_external:
            # External product stats
            col1, col2 = st.columns(2)
            
            with col1:
                # Character length distribution
                external_df["char_length"] = external_df["PRODUCT_NAME"].str.len()
                st.write("Character Length Distribution:")
                st.bar_chart(external_df["char_length"].value_counts().sort_index())
            
            with col2:
                # Word count distribution
                external_df["word_count"] = external_df["PRODUCT_NAME"].str.split().str.len()
                st.write("Word Count Distribution:")
                st.bar_chart(external_df["word_count"].value_counts().sort_index())
        
        with tab_internal:
            # Sample for performance
            sample_df = internal_df.sample(min(1000, len(internal_df)), random_state=42)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Character length distribution
                sample_df["char_length"] = sample_df["LONG_NAME"].str.len()
                st.write("Character Length Distribution (Sample):")
                st.bar_chart(sample_df["char_length"].value_counts().sort_index())
            
            with col2:
                # Word count distribution
                sample_df["word_count"] = sample_df["LONG_NAME"].str.split().str.len()
                st.write("Word Count Distribution (Sample):")
                st.bar_chart(sample_df["word_count"].value_counts().sort_index())

# Run the app
if __name__ == "__main__":
    main()