import pandas as pd
import numpy as np
import os
import time
import logging
import json
from typing import List, Dict, Tuple, Any, Optional
import re
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("product_matcher.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ProductMatcher")

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load external and internal product data from CSV files."""
    logger.info("Loading data from CSV files...")
    
    # Load external product data
    external_df = pd.read_csv("Data_External.csv")
    external_df["PRODUCT_NAME"] = external_df["PRODUCT_NAME"].str.replace("\n", "").str.strip()
    
    # Load internal product data
    internal_df = pd.read_csv("Data_Internal.csv")
    internal_df["NAME"] = internal_df["NAME"].str.replace("\n", "").str.strip()
    internal_df["OCS_NAME"] = internal_df["OCS_NAME"].str.replace("\n", "").str.strip()
    internal_df["LONG_NAME"] = internal_df["LONG_NAME"].str.replace("\n", "").str.strip()
    
    return external_df, internal_df

def load_validation_data() -> pd.DataFrame:
    """Load validation data from CSV file."""
    logger.info("Loading validation data from CSV file...")
    
    validation_df = pd.read_csv("validation_set.csv")
    validation_df["External_Product_Name"] = validation_df["External_Product_Name"].str.replace("\n", "").str.strip()
    validation_df["Valid_Internal_Product_Name"] = validation_df["Valid_Internal_Product_Name"].str.replace("\n", "").str.strip()
    
    return validation_df

def normalize_product_name(name: str) -> str:
    """Normalize product name for better matching."""
    # Convert to lowercase
    normalized = name.lower()
    
    # Replace abbreviations
    normalized = normalized.replace("w/", "with ")
    normalized = normalized.replace("&", "and")
    
    # Extract size information
    size = ""
    size_match = re.search(r"(\d+(\.\d+)?)\s*(oz|lb|g|ml|l)", normalized, re.IGNORECASE) or \
                re.search(r"\((\d+(\.\d+)?)(oz|lb|g|ml|l)\)", normalized, re.IGNORECASE)
    
    if size_match:
        # Extract just the number and unit
        size_value = size_match.group(1) or size_match.group(2)
        size_unit = (size_match.group(3) or size_match.group(4) or "").lower()
        size = f"{size_value}{size_unit}"
        
        # Remove the size from the string for now
        normalized = normalized.replace(size_match.group(0), "")
    
    # Remove any remaining parentheses, hyphens, extra spaces
    normalized = re.sub(r"[\(\)\-\.]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    
    # Add the normalized size back at the end if it exists
    if size:
        normalized = f"{normalized} {size}"
    
    return normalized

def get_openai_embeddings(batch_size: int = 10000, model_name: str = "text-embedding-3-small") -> OpenAIEmbeddings:
    """Initialize OpenAI embeddings with API key.
    
    Args:
        batch_size: Number of texts to process in each batch
        model_name: Name of the OpenAI embedding model to use
        
    Returns:
        OpenAIEmbeddings instance
    """
    # In a real implementation, the API key would be retrieved from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not found in environment variables")
        api_key = "dummy_key"  # This will be handled in the UI
    
    return OpenAIEmbeddings(batch_size=batch_size, model=model_name)

def create_vector_store(data: pd.DataFrame, column: str, embedding_model: str = "text-embedding-3-small") -> FAISS:
    """Create FAISS vector store from product names."""
    logger.info(f"Creating vector store from {len(data)} products using model {embedding_model}...")
    
    # Get normalized product names
    texts = [normalize_product_name(name) for name in data[column].tolist()]
    
    # Initialize embeddings
    embeddings = get_openai_embeddings(model_name=embedding_model)
    
    # Create vector store
    vector_store = FAISS.from_texts(texts, embeddings, metadatas=[{"id": i, "original": row} for i, row in enumerate(data[column].tolist())])
    
    return vector_store

def get_llm(model_name: str, temperature: float) -> ChatOpenAI:
    """Initialize LLM with specified model and temperature."""
    logger.info(f"Initializing LLM with model={model_name}, temperature={temperature}")
    return ChatOpenAI(model_name=model_name, temperature=temperature)

def generate_product_matching_prompt(external_product: str, candidate_products: List[str]) -> str:
    """Generate a prompt for product matching."""
    prompt = f"""
You are an expert at matching product descriptions from different systems. You need to identify if the external product matches any of the internal products.

EXTERNAL PRODUCT: "{external_product}"

CANDIDATE INTERNAL PRODUCTS:
{json.dumps(candidate_products, indent=2)}

Instructions:
1. The match has to be exact, meaning the product manufacturer, name, and size must be identical.
2. Pay special attention to size information (oz, lb, etc.) - it must match exactly.
3. Some common transformations to consider:
   - Abbreviations: "W/" in external might be "with" in internal
   - Symbols: "&" in external might be "and" in internal
   - Case differences: External products are typically in UPPERCASE
   - Size format: External might use "20 OZ" while internal uses "(20oz)"
   - Word order might differ, especially for terms like "Diet" (e.g., "DIET COKE" vs "Coca-Cola Diet")
   
4. Look for subtle differences that might indicate a non-match:
   - Different product sizes (e.g., 1.6oz vs. 1.85oz)
   - Different flavors or variants
   - Different brand names

Your task:
1. Evaluate if the external product matches any of the internal products.
2. Return your response as a JSON object with these fields:
   - "match_found": true/false
   - "matched_product": the exact internal product string that matches (or null if no match)
   - "confidence_score": a value between 0 and 100 indicating your confidence in the match
   - "reasoning": a brief explanation of your decision

IMPORTANT:
- If the sizes don't match exactly (e.g., 1.6oz vs 1.85oz), it's NOT a match.
- Brand/manufacturer must match exactly.
- Return only the JSON object with no additional text.
"""
    return prompt

def match_products(
    external_df: pd.DataFrame,
    internal_df: pd.DataFrame,
    vector_store: FAISS,
    model_name: str = "gpt-3.5-turbo",
    temperature: float = 0.0,
    confidence_threshold: float = 70.0,
    top_k: int = 5
) -> pd.DataFrame:
    """Match external products with internal products using LLM."""
    logger.info(f"Matching {len(external_df)} external products with {len(internal_df)} internal products...")
    
    # Initialize LLM
    llm = get_llm(model_name, temperature)
    
    # Initialize results DataFrame
    results = []
    
    for _, row in external_df.iterrows():
        external_product = row["PRODUCT_NAME"]
        normalized_external = normalize_product_name(external_product)
        
        logger.info(f"Processing external product: {external_product}")
        
        # Get top k similar internal products using vector similarity
        similar_docs = vector_store.similarity_search_with_score(
            normalized_external, 
            k=top_k
        )
        
        # Extract candidate internal products
        candidate_products = [internal_df.iloc[doc[0].metadata["id"]]["LONG_NAME"] for doc in similar_docs]
        candidate_scores = [doc[1] for doc in similar_docs]
        
        logger.info(f"Found {len(candidate_products)} similar internal products")
        
        # Generate prompt for LLM
        prompt = generate_product_matching_prompt(external_product, candidate_products)
        
        # Query LLM
        messages = [
            SystemMessage(content="You are an expert product matcher assistant."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = llm(messages)
            response_text = response.content
            
            # Parse JSON response
            try:
                response_json = json.loads(response_text)
                
                match_found = response_json.get("match_found", False)
                matched_product = response_json.get("matched_product", None)
                confidence_score = response_json.get("confidence_score", 0.0)
                reasoning = response_json.get("reasoning", "")
                
                # Apply confidence threshold
                if confidence_score < confidence_threshold:
                    match_found = False
                    matched_product = None
                
                result = {
                    "External_Product_Name": external_product,
                    "Matched_Internal_Product": matched_product if match_found else None,
                    "Confidence_Score": confidence_score,
                    "Match_Found": match_found,
                    "Reasoning": reasoning,
                    "Top_Candidates": candidate_products,
                    "Vector_Similarity_Scores": candidate_scores
                }
                
                results.append(result)
                logger.info(f"Match result: {result['Match_Found']}, Confidence: {result['Confidence_Score']}")
                
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response as JSON: {response_text}")
                result = {
                    "External_Product_Name": external_product,
                    "Matched_Internal_Product": None,
                    "Confidence_Score": 0.0,
                    "Match_Found": False,
                    "Reasoning": "Failed to parse LLM response",
                    "Top_Candidates": candidate_products,
                    "Vector_Similarity_Scores": candidate_scores
                }
                results.append(result)
                
        except Exception as e:
            logger.error(f"Error querying LLM: {str(e)}")
            result = {
                "External_Product_Name": external_product,
                "Matched_Internal_Product": None,
                "Confidence_Score": 0.0,
                "Match_Found": False,
                "Reasoning": f"Error querying LLM: {str(e)}",
                "Top_Candidates": candidate_products,
                "Vector_Similarity_Scores": candidate_scores
            }
            results.append(result)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df

def evaluate_validation_set(
    results_df: pd.DataFrame,
    validation_df: pd.DataFrame
) -> Dict[str, Any]:
    """Evaluate the model's performance on the validation set."""
    logger.info("Evaluating model performance on validation set...")
    
    # Merge results with validation data
    merged_df = pd.merge(
        results_df,
        validation_df,
        on="External_Product_Name",
        how="inner"
    )
    
    # Calculate metrics
    total = len(merged_df)
    correct_matches = 0
    incorrect_matches = 0
    missed_matches = 0
    correct_non_matches = 0
    incorrect_non_matches = 0
    
    for _, row in merged_df.iterrows():
        has_valid_match = row["Valid_Internal_Product_Name"] != ""
        predicted_match = row["Match_Found"]
        
        if has_valid_match and predicted_match:
            if row["Matched_Internal_Product"] == row["Valid_Internal_Product_Name"]:
                correct_matches += 1
            else:
                incorrect_matches += 1
        elif has_valid_match and not predicted_match:
            missed_matches += 1
        elif not has_valid_match and not predicted_match:
            correct_non_matches += 1
        else:  # not has_valid_match and predicted_match
            incorrect_non_matches += 1
    
    # Calculate metrics
    accuracy = (correct_matches + correct_non_matches) / total if total > 0 else 0
    precision = correct_matches / (correct_matches + incorrect_matches + incorrect_non_matches) if (correct_matches + incorrect_matches + incorrect_non_matches) > 0 else 0
    recall = correct_matches / (correct_matches + missed_matches) if (correct_matches + missed_matches) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        "total": total,
        "correct_matches": correct_matches,
        "incorrect_matches": incorrect_matches,
        "missed_matches": missed_matches,
        "correct_non_matches": correct_non_matches,
        "incorrect_non_matches": incorrect_non_matches,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
    logger.info(f"Evaluation metrics: {metrics}")
    return metrics