import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
import re

class ProductMatchingDashboard:
    """Component for visualizing product matching results in Streamlit."""
    
    def __init__(self, results_df: pd.DataFrame, validation_metrics: Dict[str, Any] = None):
        self.results_df = results_df
        self.validation_metrics = validation_metrics
    
    def render_overview(self):
        """Render the overview section of the dashboard."""
        st.header("📊 Matching Overview")
        
        # Get summary metrics
        matched_count = self.results_df["Match_Found"].sum()
        no_match_count = len(self.results_df) - matched_count
        total_count = len(self.results_df)
        match_rate = matched_count / total_count if total_count > 0 else 0
        
        # Calculate average confidence for matches and non-matches
        avg_match_confidence = self.results_df.loc[self.results_df["Match_Found"], "Confidence_Score"].mean() if matched_count > 0 else 0
        avg_no_match_confidence = self.results_df.loc[~self.results_df["Match_Found"], "Confidence_Score"].mean() if no_match_count > 0 else 0
        
        # Create metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Products", total_count)
        col2.metric("Matched Products", int(matched_count))
        col3.metric("No Match Products", int(no_match_count))
        col4.metric("Match Rate", f"{match_rate:.2%}")
        col5.metric("Avg. Match Confidence", f"{avg_match_confidence:.1f}%")
        
        # Create confidence distribution charts
        st.subheader("Confidence Score Distribution")
        
        # Create two columns for the charts
        conf_col1, conf_col2 = st.columns(2)
        
        with conf_col1:
            # Confidence distribution for matches
            if matched_count > 0:
                match_confidence_df = self.results_df[self.results_df["Match_Found"]].copy()
                
                # Create bins for confidence scores
                bins = list(range(50, 105, 5))  # 50-55, 55-60, ..., 95-100
                match_confidence_df["Confidence Bin"] = pd.cut(
                    match_confidence_df["Confidence_Score"], 
                    bins=bins, 
                    labels=[f"{i}-{i+5}" for i in range(50, 100, 5)]
                )
                
                # Count products by confidence bin
                match_confidence_counts = match_confidence_df["Confidence Bin"].value_counts().reset_index()
                match_confidence_counts.columns = ["Confidence Range", "Count"]
                match_confidence_counts = match_confidence_counts.sort_values("Confidence Range")
                
                # Create bar chart
                fig = px.bar(
                    match_confidence_counts,
                    x="Confidence Range",
                    y="Count",
                    color="Count",
                    color_continuous_scale="Blues",
                    title="Match Confidence Distribution",
                )
                fig.update_layout(xaxis_title="Confidence Score Range", yaxis_title="Number of Products")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No matches found to display confidence distribution.")
        
        with conf_col2:
            # Confidence distribution for non-matches
            if no_match_count > 0:
                no_match_confidence_df = self.results_df[~self.results_df["Match_Found"]].copy()
                
                # Create bins for confidence scores
                bins = list(range(50, 105, 5))  # 50-55, 55-60, ..., 95-100
                no_match_confidence_df["Confidence Bin"] = pd.cut(
                    no_match_confidence_df["Confidence_Score"], 
                    bins=bins, 
                    labels=[f"{i}-{i+5}" for i in range(50, 100, 5)]
                )
                
                # Count products by confidence bin
                no_match_confidence_counts = no_match_confidence_df["Confidence Bin"].value_counts().reset_index()
                no_match_confidence_counts.columns = ["Confidence Range", "Count"]
                no_match_confidence_counts = no_match_confidence_counts.sort_values("Confidence Range")
                
                # Create bar chart
                fig = px.bar(
                    no_match_confidence_counts,
                    x="Confidence Range",
                    y="Count",
                    color="Count",
                    color_continuous_scale="Reds",
                    title="No Match Confidence Distribution",
                )
                fig.update_layout(xaxis_title="Confidence Score Range", yaxis_title="Number of Products")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No non-matches found to display confidence distribution.")
    
    def render_match_details(self):
        """Render the match details section of the dashboard."""
        st.header("📋 Match Details")
        
        # Create a searchable table
        st.dataframe(
            self.results_df[["External_Product_Name", "Matched_Internal_Product", "Confidence_Score", "Match_Found", "Reasoning"]],
            hide_index=True,
            column_config={
                "External_Product_Name": "External Product",
                "Matched_Internal_Product": "Internal Product",
                "Confidence_Score": st.column_config.NumberColumn(
                    "Confidence",
                    format="%.1f%%",
                ),
                "Match_Found": "Match Found",
                "Reasoning": st.column_config.TextColumn(
                    "Reasoning",
                    width="large",
                ),
            },
        )
    
    def render_validation_results(self, validation_df: pd.DataFrame):
        """Render the validation results section."""
        if self.validation_metrics is None:
            st.warning("No validation metrics available.")
            return
        
        st.header("🔍 Validation Results")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{self.validation_metrics['accuracy']:.2%}")
        col2.metric("Precision", f"{self.validation_metrics['precision']:.2%}")
        col3.metric("Recall", f"{self.validation_metrics['recall']:.2%}")
        col4.metric("F1 Score", f"{self.validation_metrics['f1']:.2%}")
        
        # Confident predictions metrics
        st.subheader("Metrics for High Confidence Predictions")
        st.write(f"Confidence threshold: {self.validation_metrics['confidence_threshold']:.1f}%")
        
        conf_col1, conf_col2, conf_col3, conf_col4 = st.columns(4)
        conf_col1.metric("Confident Accuracy", f"{self.validation_metrics['confident_accuracy']:.2%}")
        conf_col2.metric("Confident Precision", f"{self.validation_metrics['confident_precision']:.2%}")
        conf_col3.metric("Confident Recall", f"{self.validation_metrics['confident_recall']:.2%}")
        conf_col4.metric("Confident F1", f"{self.validation_metrics['confident_f1']:.2%}")
        
        # Confusion matrix visualization
        st.subheader("Confusion Matrix")
        
        confusion_data = [
            ["Predicted Match", "Actual Match", self.validation_metrics["correct_matches"]],
            ["Predicted No Match", "Actual Match", self.validation_metrics["incorrect_non_matches"]],
            ["Predicted Match", "Actual No Match", self.validation_metrics["incorrect_matches"]],
            ["Predicted No Match", "Actual No Match", self.validation_metrics["correct_non_matches"]],
        ]
        confusion_df = pd.DataFrame(confusion_data, columns=["Predicted", "Actual", "Count"])
        
        # Create heatmap
        fig = px.density_heatmap(
            confusion_df,
            x="Actual",
            y="Predicted",
            z="Count",
            color_continuous_scale="Blues",
            text_auto=True,
        )
        fig.update_layout(
            xaxis_title="Actual",
            yaxis_title="Predicted",
            title="Confusion Matrix",
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Validation results table
        st.subheader("Validation Details")
        
        # Make a copy of validation dataframe and fill empty values with "No Match"
        validation_df_copy = validation_df.copy()
        validation_df_copy["Valid_Internal_Product_Name"] = validation_df_copy["Valid_Internal_Product_Name"].fillna("No Match")
        validation_df_copy["Valid_Internal_Product_Name"] = validation_df_copy["Valid_Internal_Product_Name"].replace("", "No Match")
        
        # Merge results with validation data
        validation_results = pd.merge(
            self.results_df,
            validation_df_copy,
            on="External_Product_Name",
            how="inner"
        )
        
        # Add match correctness column
        validation_results["Match_Correct"] = (
            (validation_results["Matched_Internal_Product"] == validation_results["Valid_Internal_Product_Name"]) | 
            ((validation_results["Matched_Internal_Product"] == "No Match") & (validation_results["Valid_Internal_Product_Name"] == "No Match"))
        )
        
        # Create table
        st.dataframe(
            validation_results[[
                "External_Product_Name",
                "Matched_Internal_Product",
                "Valid_Internal_Product_Name",
                "Confidence_Score",
                "Match_Correct"
            ]],
            hide_index=True,
            column_config={
                "External_Product_Name": "External Product",
                "Matched_Internal_Product": "Matched Internal Product",
                "Valid_Internal_Product_Name": "Valid Internal Product",
                "Confidence_Score": st.column_config.NumberColumn(
                    "Confidence",
                    format="%.1f%%",
                ),
                "Match_Correct": st.column_config.CheckboxColumn(
                    "Correct Match",
                ),
            },
        )
    
    def render_example_match(self):
        """Render an example match with detailed explanation."""
        st.header("🔎 Example Match Analysis")
        
        # Show both match and no-match examples
        match_tab, no_match_tab = st.tabs(["Match Example", "No Match Example"])
        
        with match_tab:
            # Find a successful match with high confidence
            example = None
            if not self.results_df.empty and self.results_df["Match_Found"].any():
                example = self.results_df[self.results_df["Match_Found"]].sort_values("Confidence_Score", ascending=False).iloc[0]
            
            if example is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("External Product")
                    st.code(example["External_Product_Name"])
                    
                    # Show top candidates
                    st.write("Top Similar Candidates:")
                    for i, (candidate, score) in enumerate(zip(example["Top_Candidates"], example["Vector_Similarity_Scores"])):
                        st.write(f"{i+1}. {candidate} (Similarity: {score:.4f})")
                
                with col2:
                    st.subheader("Matched Internal Product")
                    st.code(example["Matched_Internal_Product"])
                    
                    st.write("Match Details:")
                    st.write(f"- Confidence Score: {example['Confidence_Score']:.1f}%")
                    st.write(f"- Reasoning: {example['Reasoning']}")
                    
                    # Normalize both product names to show the transformation
                    st.write("Normalized Product Names:")
                    st.code(f"External: {self._normalize_for_display(example['External_Product_Name'])}")
                    st.code(f"Internal: {self._normalize_for_display(example['Matched_Internal_Product'])}")
            else:
                st.info("No matches found to display an example.")
        
        with no_match_tab:
            # Find a non-match with high confidence
            no_match_example = None
            if not self.results_df.empty and (~self.results_df["Match_Found"]).any():
                no_match_example = self.results_df[~self.results_df["Match_Found"]].sort_values("Confidence_Score", ascending=False).iloc[0]
            
            if no_match_example is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("External Product")
                    st.code(no_match_example["External_Product_Name"])
                    
                    # Show top candidates
                    st.write("Top Similar Candidates:")
                    for i, (candidate, score) in enumerate(zip(no_match_example["Top_Candidates"], no_match_example["Vector_Similarity_Scores"])):
                        st.write(f"{i+1}. {candidate} (Similarity: {score:.4f})")
                
                with col2:
                    st.subheader("Decision: No Match")
                    st.write(f"Confidence Score: {no_match_example['Confidence_Score']:.1f}%")
                    st.write(f"Reasoning: {no_match_example['Reasoning']}")
                    
                    # Normalize product name to show the transformation
                    st.write("Normalized Product Name:")
                    st.code(f"External: {self._normalize_for_display(no_match_example['External_Product_Name'])}")
            else:
                st.info("No non-matches found to display an example.")
    
    def _normalize_for_display(self, name: str) -> str:
        """Normalize product name for display purposes."""
        # Convert to lowercase
        normalized = name.lower()
        
        # Replace abbreviations
        normalized = normalized.replace("w/", "with ")
        normalized = normalized.replace("&", "and")
        
        # Highlight size information
        size_pattern = r"(\d+(\.\d+)?)\s*(oz|lb|g|ml|l)"
        size_paren_pattern = r"\((\d+(\.\d+)?)(oz|lb|g|ml|l)\)"
        
        if re.search(size_pattern, normalized, re.IGNORECASE):
            normalized = re.sub(
                size_pattern,
                lambda m: f"[{m.group(0)}]",
                normalized,
                flags=re.IGNORECASE
            )
        
        if re.search(size_paren_pattern, normalized, re.IGNORECASE):
            normalized = re.sub(
                size_paren_pattern,
                lambda m: f"[{m.group(0)}]",
                normalized,
                flags=re.IGNORECASE
            )
        
        return normalized