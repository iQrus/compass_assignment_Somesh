# AI Product Matcher

## Overview

This application automates the process of matching external product descriptions with internal product listings for a convenience store retailer. It uses a combination of:

- **Vector Embeddings**: To quickly find similar products based on semantic meaning
- **Large Language Models (LLMs)**: To make intelligent matching decisions with high accuracy
- **Prompt Engineering**: To guide the LLM in making exact product matches

## Features

- **Intelligent Matching**: Uses OpenAI's LLMs to understand product descriptions and make accurate matches
- **Vector Search**: Employs embeddings to find the most similar products efficiently
- **Confidence Scoring**: Provides a confidence score for each match to help identify uncertain cases
- **Validation**: Evaluates the system against a validation set with known matches
- **Interactive Dashboard**: Beautiful visualization of matching results and analytics
- **Customizable Parameters**: Adjust model parameters, confidence thresholds, and more from the UI

## Technical Architecture

The solution follows this pipeline:

1. **Data Loading & Preprocessing**: Clean and normalize product descriptions
2. **Embedding Generation**: Create vector embeddings for all internal products using OpenAI embeddings
3. **Vector Search**: For each external product, find top K similar internal products
4. **LLM-Based Matching**: Use a carefully engineered prompt to guide the LLM in making exact matches
5. **Confidence Scoring**: Generate a confidence score for each match
6. **Validation**: Compare results against known matches for evaluation
7. **Visualization**: Present results in an interactive dashboard

## Technologies Used

- **LangChain**: Framework for working with LLMs
- **OpenAI Embeddings**: For generating vector embeddings of product descriptions
- **FAISS**: Vector database for efficient similarity search
- **Streamlit**: For creating the interactive web application
- **Plotly**: For interactive data visualizations
- **Pandas**: For data manipulation and analysis

## Requirements

- Python 3.8+
- OpenAI API key
- Required Python packages (see requirements.txt)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/product-matcher.git
cd product-matcher
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```
   Or you can enter it directly in the application when prompted.

## Usage

1. Place your data files in the root directory:
   - `Data_External.csv`: External product list
   - `Data_Internal.csv`: Internal product list
   - `validation_set.csv`: Validation dataset (optional)

2. Run the Streamlit application:
```bash
streamlit run app.py
```

3. The web interface will open in your browser. From there you can:
   - Adjust model parameters (model, temperature, confidence threshold)
   - Run the matching process
   - Explore the results and analytics
   - Validate the system's performance

## Project Structure

```
product-matcher/
├── app.py                  # Main Streamlit application
├── product_matcher.py      # Core matching logic
├── dashboard.py            # Dashboard visualization components
├── Data_External.csv       # External product data
├── Data_Internal.csv       # Internal product data
├── validation_set.csv      # Validation dataset
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## How the Matching Works

The matching system combines vector similarity search with LLM-based matching:

1. **Vector Similarity Search**:
   - Creates embeddings for all internal products using OpenAI embeddings
   - For each external product, finds the top K most similar internal products
   - This narrows down the possibilities from 16,000+ products to just the K most promising candidates

2. **LLM-Based Matching**:
   - A carefully engineered prompt guides the LLM to evaluate if any of the candidate products matches the external product exactly
   - The prompt includes specific instructions about exact matching criteria (manufacturer, name, size)
   - The LLM returns a structured response with match decision, confidence, and reasoning

3. **Confidence Thresholding**:
   - A confidence threshold is applied to filter out uncertain matches
   - Only matches above the threshold are considered valid

## Prompt Engineering

The prompt is a critical component of the system. Key aspects include:

- Clear instructions about the exact matching criteria
- Examples of transformations to consider (abbreviations, word order, etc.)
- Specific guidance on handling size information
- Instructions to return structured output (JSON) with confidence scores and reasoning

## Validation and Metrics

The system evaluates its performance using:

- **Accuracy**: Overall correctness rate
- **Precision**: Ratio of correct matches to all predicted matches
- **Recall**: Ratio of correct matches to all actual matches
- **F1 Score**: Harmonic mean of precision and recall

## Future Improvements

Potential enhancements for the system:

1. **Training a Custom Model**: Fine-tune a model specifically for product matching
2. **Advanced Preprocessing**: Implement more sophisticated normalization techniques
3. **User Feedback Loop**: Allow users to correct mistakes and use that to improve future matches
4. **Batch Processing**: Add support for batch matching of large product lists
5. **Additional Data Sources**: Incorporate additional product data (e.g., descriptions, categories) for better matching
