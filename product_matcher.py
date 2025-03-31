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
    level=logging.INFO,
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

def get_openai_embeddings(batch_size: int = 2000, model_name: str = "text-embedding-3-small") -> OpenAIEmbeddings:
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
    
    return OpenAIEmbeddings(chunk_size=batch_size, model=model_name)

def create_vector_store(data: pd.DataFrame, column: str, embedding_model: str = "text-embedding-3-small") -> FAISS:
    """Create FAISS vector store from product names.
    
    Args:
        data: DataFrame containing product data
        column: Column name containing product names
        embedding_model: Name of the embedding model to use
        
    Returns:
        FAISS vector store
    """
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

def match_products_batch(
    external_df: pd.DataFrame,
    internal_df: pd.DataFrame,
    vector_store: FAISS,
    model_name: str = "gpt-4",
    temperature: float = 0.0,
    confidence_threshold: float = 70.0,
    top_k: int = 5,
    batch_size: int = 5
) -> pd.DataFrame:
    """Match external products with internal products in batches using LLM.
    
    Args:
        external_df: DataFrame containing external products
        internal_df: DataFrame containing internal products
        vector_store: FAISS vector store of internal products
        model_name: Name of the LLM model to use
        temperature: Temperature for LLM generation
        confidence_threshold: Minimum confidence score for a match
        top_k: Number of candidate products to consider
        batch_size: Number of products to process in each batch
        
    Returns:
        DataFrame with matching results
    """
    logger.info(f"Matching {len(external_df)} external products with {len(internal_df)} internal products in batches of {batch_size}...")
    
    # Initialize LLM
    llm = get_llm(model_name, temperature)
    
    # Initialize results list
    results = []
    
    # Process in batches
    for i in range(0, len(external_df), batch_size):
        batch_df = external_df.iloc[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1} with {len(batch_df)} products (products {i+1}-{min(i+batch_size, len(external_df))})")
        
        batch_results = []
        batch_candidates = []
        
        # Prepare data for each product in the batch
        for _, row in batch_df.iterrows():
            external_product = row["PRODUCT_NAME"]
            normalized_external = normalize_product_name(external_product)
            
            # Get top k similar internal products using vector similarity
            similar_docs = vector_store.similarity_search_with_score(
                normalized_external, 
                k=top_k
            )
            
            # Extract candidate internal products
            candidate_products = [internal_df.iloc[doc[0].metadata["id"]]["LONG_NAME"] for doc in similar_docs]
            candidate_scores = [doc[1] for doc in similar_docs]
            
            batch_candidates.append({
                "external_product": external_product,
                "candidate_products": candidate_products,
                "candidate_scores": candidate_scores
            })
        
        # Generate batch prompt for LLM
        batch_prompt = generate_batch_matching_prompt(batch_candidates)
        
        # Query LLM with batch prompt
        messages = [
            SystemMessage(content="You are an expert product matcher assistant."),
            HumanMessage(content=batch_prompt)
        ]
        
        try:
            response = llm(messages)
            response_text = response.content
            
            # Parse JSON response
            try:
                response_json = json.loads(response_text)
                batch_results = response_json.get("results", [])
                
                # Process each result in the batch
                for j, result_item in enumerate(batch_results):
                    if j >= len(batch_candidates):
                        logger.warning(f"Received more results than expected in batch {i//batch_size + 1}")
                        break
                        
                    external_product = batch_candidates[j]["external_product"]
                    candidate_products = batch_candidates[j]["candidate_products"]
                    candidate_scores = batch_candidates[j]["candidate_scores"]
                    
                    match_found = result_item.get("match_found", False)
                    matched_product = result_item.get("matched_product", None)
                    confidence_score = result_item.get("confidence_score", 0.0)
                    reasoning = result_item.get("reasoning", "")
                    
                    # Apply confidence threshold for positive matches
                    if match_found and confidence_score < confidence_threshold:
                        match_found = False
                        matched_product = "No Match"
                    elif not match_found:
                        matched_product = "No Match"
                    
                    result = {
                        "External_Product_Name": external_product,
                        "Matched_Internal_Product": matched_product,
                        "Confidence_Score": confidence_score,
                        "Match_Found": match_found,
                        "Reasoning": reasoning,
                        "Top_Candidates": candidate_products,
                        "Vector_Similarity_Scores": candidate_scores
                    }
                    
                    results.append(result)
                    logger.info(f"Match result for {external_product}: {match_found}, Confidence: {confidence_score}")
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON in batch {i//batch_size + 1}: {str(e)}")
                logger.error(f"Response text: {response_text}")
                
                # Handle the error by processing each product individually as fallback
                for item in batch_candidates:
                    external_product = item["external_product"]
                    candidate_products = item["candidate_products"]
                    candidate_scores = item["candidate_scores"]
                    
                    result = {
                        "External_Product_Name": external_product,
                        "Matched_Internal_Product": "No Match",
                        "Confidence_Score": 0.0,
                        "Match_Found": False,
                        "Reasoning": f"Error parsing batch response: {str(e)}",
                        "Top_Candidates": candidate_products,
                        "Vector_Similarity_Scores": candidate_scores
                    }
                    results.append(result)
                
        except Exception as e:
            logger.error(f"Error querying LLM for batch {i//batch_size + 1}: {str(e)}")
            
            # Handle the error by adding error results for all products in the batch
            for item in batch_candidates:
                external_product = item["external_product"]
                candidate_products = item["candidate_products"]
                candidate_scores = item["candidate_scores"]
                
                result = {
                    "External_Product_Name": external_product,
                    "Matched_Internal_Product": "No Match",
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

def generate_batch_matching_prompt(batch_candidates: List[Dict[str, Any]]) -> str:
    """Generate a prompt for batch product matching.
    
    Args:
        batch_candidates: List of dictionaries containing external products and their candidates
        
    Returns:
        Prompt string for batch matching
    """
    prompt = """
You are an expert at matching product descriptions from different systems. You need to identify if each external product matches any of the corresponding internal product candidates.

I will provide you with multiple external products and their candidate internal matches. For each external product, evaluate if it matches any of the candidate internal products.

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

PRODUCT MATCHING TASKS:
"""

    for i, item in enumerate(batch_candidates):
        external_product = item["external_product"]
        candidate_products = item["candidate_products"]
        
        prompt += f"""
Task {i+1}:
EXTERNAL PRODUCT: "{external_product}"

CANDIDATE INTERNAL PRODUCTS:
{json.dumps(candidate_products, indent=2)}

"""

    prompt += """
Your task:
1. For EACH external product, evaluate if it matches any of its candidate internal products.
2. Return your response as a JSON object with this structure:
{
  "results": [
    {
      "match_found": true/false,
      "matched_product": "the exact internal product string that matches (or null if no match)",
      "confidence_score": 0-100,
      "reasoning": "brief explanation"
    },
    // Additional results for each product...
  ]
}

IMPORTANT:
- Always provide a confidence score even if no match is found. For non-matches, the confidence score should reflect how certain you are that no match exists.
- If the sizes don't match exactly (e.g., 1.6oz vs 1.85oz), it's NOT a match.
- Brand/manufacturer must match exactly.
- Return ONLY the JSON object with no additional text.
- Ensure you return results for ALL products in the order they were presented.
"""
    return prompt

def evaluate_validation_set(
    results_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    confidence_threshold: float = 70.0
) -> Dict[str, Any]:
    """Evaluate the model's performance on the validation set."""
    logger.info("Evaluating model performance on validation set...")
    
    # Make a copy of validation dataframe and fill empty values with "No Match"
    validation_df_copy = validation_df.copy()
    validation_df_copy["Valid_Internal_Product_Name"] = validation_df_copy["Valid_Internal_Product_Name"].fillna("No Match")
    validation_df_copy["Valid_Internal_Product_Name"] = validation_df_copy["Valid_Internal_Product_Name"].replace("", "No Match")
    
    # Merge results with validation data
    merged_df = pd.merge(
        results_df,
        validation_df_copy,
        on="External_Product_Name",
        how="inner"
    )
    
    # Calculate metrics
    total = len(merged_df)
    correct_matches = 0
    incorrect_matches = 0
    correct_non_matches = 0
    incorrect_non_matches = 0
    confident_correct_matches = 0
    confident_incorrect_matches = 0
    confident_correct_non_matches = 0
    confident_incorrect_non_matches = 0
    
    for _, row in merged_df.iterrows():
        has_valid_match = row["Valid_Internal_Product_Name"] != "No Match"
        predicted_match = row["Match_Found"]
        is_confident = row["Confidence_Score"] >= confidence_threshold
        
        if has_valid_match and predicted_match:
            if row["Matched_Internal_Product"] == row["Valid_Internal_Product_Name"]:
                correct_matches += 1
                if is_confident:
                    confident_correct_matches += 1
            else:
                incorrect_matches += 1
                if is_confident:
                    confident_incorrect_matches += 1
        elif has_valid_match and not predicted_match:
            incorrect_non_matches += 1
            if is_confident:
                confident_incorrect_non_matches += 1
        elif not has_valid_match and not predicted_match:
            correct_non_matches += 1
            if is_confident:
                confident_correct_non_matches += 1
        else:  # not has_valid_match and predicted_match
            incorrect_matches += 1
            if is_confident:
                confident_incorrect_matches += 1
    
    # Calculate overall metrics
    accuracy = (correct_matches + correct_non_matches) / total if total > 0 else 0
    precision = correct_matches / (correct_matches + incorrect_matches) if (correct_matches + incorrect_matches) > 0 else 0
    recall = correct_matches / (correct_matches + incorrect_non_matches) if (correct_matches + incorrect_non_matches) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate metrics for confident predictions
    confident_total = confident_correct_matches + confident_incorrect_matches + confident_correct_non_matches + confident_incorrect_non_matches
    confident_accuracy = (confident_correct_matches + confident_correct_non_matches) / confident_total if confident_total > 0 else 0
    confident_precision = confident_correct_matches / (confident_correct_matches + confident_incorrect_matches) if (confident_correct_matches + confident_incorrect_matches) > 0 else 0
    confident_recall = confident_correct_matches / (confident_correct_matches + confident_incorrect_non_matches) if (confident_correct_matches + confident_incorrect_non_matches) > 0 else 0
    confident_f1 = 2 * confident_precision * confident_recall / (confident_precision + confident_recall) if (confident_precision + confident_recall) > 0 else 0
    
    metrics = {
        "total": total,
        "correct_matches": correct_matches,
        "incorrect_matches": incorrect_matches,
        "correct_non_matches": correct_non_matches,
        "incorrect_non_matches": incorrect_non_matches,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confident_total": confident_total,
        "confident_correct_matches": confident_correct_matches,
        "confident_incorrect_matches": confident_incorrect_matches,
        "confident_correct_non_matches": confident_correct_non_matches,
        "confident_incorrect_non_matches": confident_incorrect_non_matches,
        "confident_accuracy": confident_accuracy,
        "confident_precision": confident_precision,
        "confident_recall": confident_recall,
        "confident_f1": confident_f1,
        "confidence_threshold": confidence_threshold
    }
    
    logger.info(f"Evaluation metrics: {metrics}")
    return metrics