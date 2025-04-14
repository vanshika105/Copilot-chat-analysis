import pandas as pd
import numpy as np
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def extract_question_type(prompt_str):
    """
    Extract the question type from the prompt JSON string.
    
    Args:
        prompt_str (str): JSON string from the Prompt column
        
    Returns:
        str: Question type based on the tool used
    """
    # Handle empty or non-JSON prompts
    if pd.isna(prompt_str) or prompt_str == "[]" or not prompt_str:
        return "No Tool Used"
    
    try:
        # Parse the JSON string
        prompt_data = json.loads(prompt_str)
        
        # If the prompt is empty list
        if not prompt_data:
            return "No Tool Used"
        
        # Extract the tool from the first action
        if prompt_data and isinstance(prompt_data, list) and len(prompt_data) > 0:
            if 'action' in prompt_data[0] and 'tool' in prompt_data[0]['action']:
                tool = prompt_data[0]['action']['tool']
                
                # Map tool names to categories
                tool_map = {
                    'shiprocket_knowledgebase': 'KB Question',
                    'order_tracking': 'Order Tracking',
                    'cod_remittance_tool': 'COD Remittance',
                    'rto_performance_tool': 'RTO Performance',
                    'HARDCODED_RESPONSE': 'Hardcoded Question',
                    'shipping_rate_calculator': 'Shipping Rate'
                }
                
                return tool_map.get(tool, tool.replace('_', ' ').title())
            
        return "Other"
    except Exception as e:
        # Handle parsing errors
        return "Invalid Format"

def process_data(df):
    """
    Process the uploaded data containing user emails and questions to create user cohorts.
    
    Args:
        df (pd.DataFrame): DataFrame containing user email and question data
        
    Returns:
        tuple: (processed_df, cohort_df) - processed data and cohort summary
    """
    # Make a copy to avoid modifying the original
    raw_df = df.copy()
    
    # Ensure the required columns exist
    required_cols = ["user_email", "question"]
    for col in required_cols:
        if col not in raw_df.columns:
            raise ValueError(f"Data must contain '{col}' column")
    
    # No need to add question_type here as we'll handle it in get_question_type_distribution
    
    # Count chats per user
    user_chat_counts = raw_df.groupby("user_email").size().reset_index(name="chat_count")
    
    # Create a processed dataframe with user_id and chat_count
    processed_df = pd.DataFrame({
        "user_email": user_chat_counts["user_email"],
        "chat_count": user_chat_counts["chat_count"]
    })
    
    # Create cohort indicators
    processed_df["cohort_gt_3"] = processed_df["chat_count"] > 3
    processed_df["cohort_gt_5"] = processed_df["chat_count"] > 5
    processed_df["cohort_gt_10"] = processed_df["chat_count"] > 10
    processed_df["cohort_gt_15"] = processed_df["chat_count"] > 15
    
    # Create cohort summary data
    thresholds = [3, 5, 10, 15]
    cohort_data = {
        "Threshold": thresholds,
        "Users": [
            sum(processed_df["chat_count"] > threshold)
            for threshold in thresholds
        ],
        "Percentage": [
            sum(processed_df["chat_count"] > threshold) / len(processed_df) * 100
            for threshold in thresholds
        ]
    }
    
    cohort_df = pd.DataFrame(cohort_data)
    
    return processed_df, cohort_df

def get_question_type_distribution(df, cohort_users):
    """
    Get the distribution of question types for a cohort of users.
    
    Args:
        df (pd.DataFrame): Original dataframe with user_email and Prompt columns
        cohort_users (list): List of user emails in the cohort
        
    Returns:
        pd.DataFrame: DataFrame with question type counts and percentages
    """
    # Filter dataframe for users in cohort
    cohort_df = df[df["user_email"].isin(cohort_users)]
    
    # Ensure we have the question_type column
    if 'question_type' not in cohort_df.columns:
        # If not, extract it from the Prompt column
        if 'Prompt' in cohort_df.columns:
            cohort_df['question_type'] = cohort_df['Prompt'].apply(extract_question_type)
        else:
            # If no Prompt column, create a default question type
            cohort_df['question_type'] = 'Unknown'
    
    # Count question types
    type_counts = cohort_df["question_type"].value_counts().reset_index()
    type_counts.columns = ["question_type", "count"]
    
    # Calculate percentages
    total = type_counts["count"].sum()
    type_counts["percentage"] = (type_counts["count"] / total * 100).round(1)
    
    return type_counts

def analyze_sentiment(text):
    """
    Analyze the sentiment of a text with special focus on frustration, anger, and dissatisfaction.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        str: Sentiment category (Positive, Negative, or Neutral)
    """
    # Initialize the VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    
    # Get basic sentiment scores
    scores = analyzer.polarity_scores(text)
    
    # Define frustration/anger/dissatisfaction indicators
    frustration_phrases = [
        "not working", "doesn't work", "doesn't help", "doesn't show", 
        "can't find", "cannot find", "unable to", "failed to",
        "keep getting", "keeps showing", "still not", "still getting",
        "why is this", "why does this", "why can't",
        "frustrated", "frustrating", "annoying", "annoyed",
        "terrible", "horrible", "awful", "worst",
        "fix this", "fix it",
        "angry", "upset", "disappointed", "dissatisfied",
        "wasting time", "waste of time", "useless", "unhelpful"
    ]
    
    question_frustration_indicators = [
        "why isn't", "why won't", "why doesn't", "why can't", 
        "how do i fix", "when will this be fixed", "is this a bug"
    ]
    
    # Phrases that are often neutral despite containing help-seeking words
    neutral_help_phrases = [
        "how can", "how do i", "how to", "can you help", "help me",
        "need help", "is there a way", "what is the best way",
        "how does", "what are", "is it possible", "can i"
    ]
    
    # Check for presence of frustration/anger/dissatisfaction phrases
    text_lower = text.lower()
    
    # Check for information-seeking or help-seeking questions that are neutral
    neutral_help_indicators = any(phrase in text_lower for phrase in neutral_help_phrases)
    
    # Only count as neutral help if there are no frustration indicators
    if neutral_help_indicators:
        # Specifically check if it's a question about help without frustration
        information_seeking = (
            "how can" in text_lower or 
            "what is" in text_lower or
            "can you" in text_lower or
            "is there" in text_lower
        )
        
        # If it's a neutral question specifically asking for help or information
        # without expressing frustration, treat it as neutral
        if information_seeking and scores['compound'] > -0.2:
            return 'Neutral'
    
    # Count frustration indicators
    frustration_count = sum(1 for phrase in frustration_phrases if phrase in text_lower)
    question_indicator_count = sum(1 for phrase in question_frustration_indicators if phrase in text_lower)
    
    # Adjust sentiment based on frustration/anger/dissatisfaction indicators
    if frustration_count >= 1 or question_indicator_count >= 1:
        # Double check that we're not misclassifying general help requests
        if "how can" in text_lower and "help" in text_lower and scores['compound'] > -0.1:
            return 'Neutral'
        # If we detect clear frustration indicators, classify as negative regardless of VADER score
        return 'Negative'
    elif scores['compound'] >= 0.05:
        return 'Positive'
    elif scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def get_sentiment_distribution(df, cohort_users=None):
    """
    Get sentiment distribution for questions in a cohort.
    
    Args:
        df (pd.DataFrame): DataFrame with user_email and question columns
        cohort_users (list, optional): List of user emails to filter by. If None, analyze all users.
        
    Returns:
        pd.DataFrame: DataFrame with sentiment counts and percentages
    """
    # Filter by cohort if specified
    if cohort_users is not None:
        data = df[df["user_email"].isin(cohort_users)].copy()
    else:
        data = df.copy()
    
    # Apply sentiment analysis to each question
    data['sentiment'] = data['question'].apply(analyze_sentiment)
    
    # Count sentiments
    sentiment_counts = data['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'count']
    
    # Calculate percentages
    total = sentiment_counts['count'].sum()
    sentiment_counts['percentage'] = (sentiment_counts['count'] / total * 100).round(1)
    
    return sentiment_counts

def calculate_cohort_metrics(df):
    """
    Calculate metrics for each chat threshold cohort.
    
    Args:
        df (pd.DataFrame): Processed DataFrame with chat counts
        
    Returns:
        dict: Dictionary containing metrics for each cohort
    """
    total_users = len(df)
    metrics = {}
    
    # Calculate metrics for each threshold
    for threshold in [3, 5, 10, 15]:
        users_above = sum(df["chat_count"] >= threshold)
        
        metrics[f"users_gte_{threshold}"] = users_above
        metrics[f"pct_gte_{threshold}"] = (users_above / total_users) * 100
        
        # Subset of users at or above threshold
        users_subset = df[df["chat_count"] >= threshold]
        
        if not users_subset.empty:
            metrics[f"avg_chats_gte_{threshold}"] = users_subset["chat_count"].mean()
            metrics[f"median_chats_gte_{threshold}"] = users_subset["chat_count"].median()
            metrics[f"max_chats_gte_{threshold}"] = users_subset["chat_count"].max()
        else:
            metrics[f"avg_chats_gte_{threshold}"] = 0
            metrics[f"median_chats_gte_{threshold}"] = 0
            metrics[f"max_chats_gte_{threshold}"] = 0
    
    return metrics
