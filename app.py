import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json
from utils import process_data, calculate_cohort_metrics, get_question_type_distribution, extract_question_type, get_sentiment_distribution, analyze_sentiment
from database import save_data_to_db, load_data_from_db, get_user_count, get_question_count

# Set page configuration
st.set_page_config(
    page_title="User Chat Cohort Analysis",
    page_icon="💬",
    layout="wide"
)

# Page title and introduction
st.title("User Chat Cohort Analysis Dashboard")
st.markdown("""
    This dashboard helps you analyze user cohorts based on chat frequency.
    Data is loaded from the database which contains user emails and their questions.
""")

# Sidebar for database controls and info
with st.sidebar:
    st.header("Database Info")
    
    # Option to load data from the database or from the CSV file directly
    data_source = st.radio(
        "Select Data Source",
        ["Database", "CSV File (attached_assets/query_result_2025-04-07T13_03_27.309441Z.csv)"]
    )
    
    if data_source == "Database":
        try:
            # Check if data exists in the database
            user_count = get_user_count()
            question_count = get_question_count()
            
            if user_count > 0:
                st.success(f"Database contains {user_count} users and {question_count} questions.")
            else:
                st.warning("Database is empty. Use the button below to import data.")
                
                # Button to populate the database from the CSV file
                if st.button("Import Data from CSV to Database"):
                    from populate_db import populate_database
                    success = populate_database()
                    if success:
                        st.success("Data successfully imported to database!")
                        st.rerun()
                    else:
                        st.error("Failed to import data to database.")
        except Exception as e:
            st.error(f"Error accessing database: {str(e)}")
    
    st.markdown("### About")
    st.info("""
        This dashboard automatically:
        1. Counts the number of chats per user
        2. Segments users into cohorts based on chat frequency:
           - Cohort 1: >= 3 chats
           - Cohort 2: >= 5 chats
           - Cohort 3: >= 10 chats
           - Cohort 4: >= 15 chats
        3. Categorizes question types from the Prompt field:
           - KB Question (from shiprocket_knowledgebase tool)
           - Order Tracking (from order_tracking tool)
           - COD Remittance (from cod_remittance_tool)
           - And other tool categories
    """)

# Main content area
try:
    # Load data based on selected source
    if data_source == "Database":
        try:
            # Attempt to load from database
            df = load_data_from_db()
            if df.empty:
                st.error("No data found in the database. Please import data first.")
                st.stop()
        except Exception as e:
            st.error(f"Error loading data from database: {str(e)}")
            st.stop()
    else:
        # Load from CSV file
        try:
            # Use the new query result dataset
            file_path = 'attached_assets/query_result_2025-04-07T13_03_27.309441Z.csv'
            df = pd.read_csv(file_path)
            
            # Rename columns to match expected format
            df = df.rename(columns={
                'Userid': 'user_email',
                'Question': 'question'
            })
            
        except Exception as e:
            st.error(f"Error loading CSV file: {str(e)}")
            st.stop()
    
    # Check if required columns exist
    required_cols = ["user_email", "question"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}. Please ensure your data contains these columns.")
    else:
        # Process data and create cohorts
        processed_df, cohort_df = process_data(df)
            
        # Display data overview
        st.header("Data Overview")
        col1, col2, col3 = st.columns(3)
            
        with col1:
            st.metric("Total Users", len(processed_df))
            st.metric("Total Chats", int(processed_df["chat_count"].sum()))
        
        with col2:
            st.metric("Average Chats per User", round(processed_df["chat_count"].mean(), 2))
            st.metric("Median Chats", processed_df["chat_count"].median())
        
        with col3:
            st.metric("Max Chats", processed_df["chat_count"].max())
            st.metric("Min Chats", processed_df["chat_count"].min())
        
        # Display raw data samples
        with st.expander("View Raw Data Sample"):
            st.dataframe(df.head(10))
        
        # Display processed data
        with st.expander("View Processed User Data"):
            st.dataframe(processed_df.head(10))
        
        # Cohort Analysis Section
        st.header("Cohort Analysis")
        
        # Cohort distribution metrics
        metrics = calculate_cohort_metrics(processed_df)
        
        # Create clickable cards for cohort metrics
        st.markdown("### Click on a cohort to see users and their chats")
        cols = st.columns(4)
        cohort_names = [">= 3 Chats", ">= 5 Chats", ">= 10 Chats", ">= 15 Chats"]
        thresholds = [3, 5, 10, 15]
            
            # Session state to track selected cohort
        if 'selected_cohort' not in st.session_state:
            st.session_state.selected_cohort = None
        
        # Create clickable cards
        for i, (col, name, threshold) in enumerate(zip(cols, cohort_names, thresholds)):
            with col:
                cohort_count = metrics[f"users_gte_{threshold}"]
                cohort_pct = metrics[f"pct_gte_{threshold}"]
                    
                # Create a clickable card with CSS
                card_html = f"""
                <div style="
                    padding: 15px; 
                    border-radius: 10px; 
                    background-color: {'#e6f3ff' if st.session_state.selected_cohort == threshold else '#f0f2f6'};
                    border: 2px solid {'#2196F3' if st.session_state.selected_cohort == threshold else '#ddd'};
                    text-align: center;
                    cursor: pointer;
                    transition: all 0.3s;
                    &:hover {{
                        background-color: #e6f3ff;
                        border-color: #2196F3;
                    }}
                ">
                    <h3 style="margin: 0; color: #333;">{name}</h3>
                    <p style="font-size: 24px; font-weight: bold; margin: 10px 0; color: #2196F3;">{cohort_count} users</p>
                    <p style="margin: 0; color: #666;">{cohort_pct:.1f}% of total</p>
                </div>
                """
                    
                # Use a button with the same dimensions as the card
                if st.button(f"Cohort {threshold}", key=f"cohort_button_{threshold}", help=f"Click to view users with {threshold} or more chats"):
                    st.session_state.selected_cohort = threshold
                    # Force a rerun to update the UI
                    st.rerun()
                
                # Display the card (replaces the button visually)
                st.markdown(card_html, unsafe_allow_html=True)
            
        # Display the selected cohort data
        if st.session_state.selected_cohort is not None:
            threshold = st.session_state.selected_cohort
            filtered_users = processed_df[processed_df["chat_count"] >= threshold]
            
            st.markdown(f"### Users with >= {threshold} chats ({len(filtered_users)} users)")
            
            # Raw chats for these users
            users_in_cohort = filtered_users["user_email"].tolist()
            
            # Get all chats for these users
            cohort_chats = df[df["user_email"].isin(users_in_cohort)]
            
            # Create tabs for different aspects of the cohort
            cohort_tab1, cohort_tab2, cohort_tab3 = st.tabs(["Users", "Chats", "Question Types"])
            
            with cohort_tab1:
                # Display user chat counts
                st.markdown("#### Users in this cohort")
                st.dataframe(filtered_users, use_container_width=True)
            
            with cohort_tab2:
                # Display chats for these users
                st.markdown("#### All chats from users in this cohort")
                st.dataframe(cohort_chats, use_container_width=True)
            
            with cohort_tab3:
                # Display question type distribution for this cohort
                st.markdown("#### Question Type Distribution")
                
                # Get question type distribution for this cohort
                question_types_df = get_question_type_distribution(df, users_in_cohort)
                
                # Display the table
                col1, col2 = st.columns([2, 3])
                
                with col1:
                    st.dataframe(question_types_df, use_container_width=True)
                
                with col2:
                    # Create pie chart of question types
                    fig = px.pie(
                        question_types_df,
                        values="count",
                        names="question_type",
                        title=f"Question Types for Cohort >= {threshold} Chats",
                        hole=0.4
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Add interactive question type selection
                st.markdown("#### View Questions by Type")
                st.write("Click on a question type to see all questions of that type:")
                
                # Filter cohort data with original df to include all columns
                cohort_data = df[df["user_email"].isin(users_in_cohort)].copy()
                
                # Make sure we have question_type column
                if 'question_type' not in cohort_data.columns:
                    if 'Prompt' in cohort_data.columns:
                        cohort_data.loc[:, 'question_type'] = cohort_data['Prompt'].apply(extract_question_type)
                    else:
                        cohort_data.loc[:, 'question_type'] = 'Unknown'
                
                # Get unique question types for selection
                question_types = cohort_data['question_type'].unique().tolist()
                
                # Create selection widget
                selected_type = st.selectbox("Select a question type", question_types)
                
                # Show questions of selected type
                if selected_type:
                    filtered_questions = cohort_data[cohort_data['question_type'] == selected_type]
                    st.write(f"**{len(filtered_questions)} questions of type '{selected_type}':**")
                    
                    # Display questions in a table
                    questions_table = filtered_questions[['user_email', 'question']].reset_index(drop=True)
                    st.dataframe(questions_table, use_container_width=True)
            
            # Clear selection button
            if st.button("Clear selection", key="clear_cohort_selection"):
                st.session_state.selected_cohort = None
                st.rerun()
            
        # Visualization section
        st.header("Visualizations")
        
        # Tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["Cohort Distribution", "Chat Frequency", "Cumulative Distribution", "Sentiment Analysis"])
            
        with tab1:
            # Bar chart showing number of users in each cohort
            cohort_data = {
                "Cohort": [f">= {threshold} Chats" for threshold in thresholds],
                "Number of Users": [metrics[f"users_gte_{threshold}"] for threshold in thresholds]
            }
            cohort_df = pd.DataFrame(cohort_data)
            
            fig = px.bar(
                cohort_df,
                x="Cohort",
                y="Number of Users",
                color="Cohort",
                text="Number of Users",
                title="Number of Users in Each Cohort"
            )
            fig.update_layout(xaxis_title="Chat Threshold", yaxis_title="Number of Users")
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            # Histogram of chat frequency distribution
            fig = px.histogram(
                processed_df,
                x="chat_count",
                nbins=50,
                title="Distribution of Chat Counts",
                labels={"chat_count": "Number of Chats", "count": "Number of Users"}
            )
            
            # Add vertical lines for the thresholds
            for threshold in thresholds:
                fig.add_vline(
                    x=threshold,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f">= {threshold}",
                    annotation_position="top right"
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
        with tab3:
            # Cumulative distribution chart
            chat_counts = processed_df["chat_count"].value_counts().sort_index()
            cumulative = 1 - chat_counts.cumsum() / chat_counts.sum()
            
            # Create DataFrame for plotting
            cumulative_df = pd.DataFrame({
                "chat_count": cumulative.index,
                "pct_users_above": cumulative.values * 100
            })
            
            fig = px.line(
                cumulative_df,
                x="chat_count",
                y="pct_users_above",
                title="Percentage of Users Above Chat Count",
                labels={
                    "chat_count": "Number of Chats",
                    "pct_users_above": "% of Users Above This Count"
                }
            )
            
            # Add markers for the thresholds
            for threshold in thresholds:
                # Find the percentage for this threshold
                pct = 0
                if threshold in cumulative_df["chat_count"].values:
                    pct = cumulative_df[cumulative_df["chat_count"] == threshold]["pct_users_above"].values[0]
                else:
                    # Interpolate if exact value not in data
                    closest_below = cumulative_df[cumulative_df["chat_count"] < threshold]["chat_count"].max()
                    closest_above = cumulative_df[cumulative_df["chat_count"] > threshold]["chat_count"].min()
                    
                    if not pd.isna(closest_below) and not pd.isna(closest_above):
                        pct_below = cumulative_df[cumulative_df["chat_count"] == closest_below]["pct_users_above"].values[0]
                        pct_above = cumulative_df[cumulative_df["chat_count"] == closest_above]["pct_users_above"].values[0]
                        
                        # Linear interpolation
                        pct = pct_below - (pct_below - pct_above) * (threshold - closest_below) / (closest_above - closest_below)
                
                fig.add_trace(
                    go.Scatter(
                        x=[threshold],
                        y=[pct],
                        mode="markers+text",
                        marker=dict(size=10, color="red"),
                        text=[f"{pct:.1f}%"],
                        textposition="top center",
                        name=f">= {threshold} chats"
                    )
                )
            
            fig.update_layout(showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
        with tab4:
            # Sentiment analysis of all questions
            st.markdown("### Sentiment Analysis of Questions")
            
            # Create tabs for different sentiment views
            sentiment_tab1, sentiment_tab2 = st.tabs(["Overall Sentiment", "Sentiment by Cohort"])
            
            with sentiment_tab1:
                # Get sentiment distribution for all questions
                sentiment_df = get_sentiment_distribution(df)
                
                # Display table and chart
                col1, col2 = st.columns([2, 3])
                
                with col1:
                    st.dataframe(sentiment_df, use_container_width=True)
                
                with col2:
                    # Create pie chart of sentiment distribution
                    colors = {'Positive': '#28a745', 'Neutral': '#ffc107', 'Negative': '#dc3545'}
                    fig = px.pie(
                        sentiment_df,
                        values="count",
                        names="sentiment",
                        title="Overall Sentiment Distribution",
                        color="sentiment",
                        color_discrete_map=colors,
                        hole=0.4
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Add interactive sentiment selection for viewing questions
                st.markdown("#### View Questions by Sentiment")
                st.write("Select a sentiment to see all questions with that sentiment:")
                
                # Make sure we have sentiment column in the dataframe
                sentiment_data = df.copy()
                if 'sentiment' not in sentiment_data.columns:
                    sentiment_data['sentiment'] = sentiment_data['question'].apply(analyze_sentiment)
                
                # Get unique sentiments
                sentiments = sentiment_data['sentiment'].unique().tolist()
                
                # Create selection widget
                selected_sentiment = st.selectbox("Select a sentiment", sentiments, key="overall_sentiment")
                
                # Show questions with selected sentiment
                if selected_sentiment:
                    filtered_questions = sentiment_data[sentiment_data['sentiment'] == selected_sentiment]
                    st.write(f"**{len(filtered_questions)} questions with '{selected_sentiment}' sentiment:**")
                    
                    # Display questions in a table
                    sentiment_table = filtered_questions[['user_email', 'question']].reset_index(drop=True)
                    st.dataframe(sentiment_table, use_container_width=True)
            
            with sentiment_tab2:
                # Show sentiment distribution for each cohort
                st.markdown("#### Sentiment Distribution by Cohort")
                
                # Create dataframe to store sentiment counts for each cohort
                cohort_sentiments = []
                
                # Calculate sentiment distribution for each cohort
                for threshold in thresholds:
                    cohort_users = processed_df[processed_df["chat_count"] >= threshold]["user_email"].tolist()
                    sentiment_counts = get_sentiment_distribution(df, cohort_users)
                    
                    # Add cohort label
                    sentiment_counts["cohort"] = f">= {threshold} Chats"
                    cohort_sentiments.append(sentiment_counts)
                
                # Combine all cohort sentiment data
                if cohort_sentiments:
                    combined_sentiments = pd.concat(cohort_sentiments)
                    
                    # Create grouped bar chart
                    fig = px.bar(
                        combined_sentiments,
                        x="cohort",
                        y="count",
                        color="sentiment",
                        barmode="group",
                        text="count",
                        title="Sentiment Distribution by Cohort",
                        color_discrete_map={'Positive': '#28a745', 'Neutral': '#ffc107', 'Negative': '#dc3545'}
                    )
                    fig.update_layout(xaxis_title="Cohort", yaxis_title="Number of Questions")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create another view showing sentiment percentages
                    fig2 = px.bar(
                        combined_sentiments,
                        x="cohort",
                        y="percentage",
                        color="sentiment",
                        barmode="group",
                        text="percentage",
                        title="Sentiment Percentage by Cohort",
                        color_discrete_map={'Positive': '#28a745', 'Neutral': '#ffc107', 'Negative': '#dc3545'}
                    )
                    fig2.update_layout(xaxis_title="Cohort", yaxis_title="Percentage (%)")
                    fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Allow selection of a sentiment to see questions in all cohorts
                    st.markdown("#### View Questions by Sentiment Across Cohorts")
                    
                    # Get all sentiments available
                    all_sentiments = combined_sentiments['sentiment'].unique().tolist()
                    
                    # Create selection widget
                    selected_cohort_sentiment = st.selectbox(
                        "Select a sentiment to view questions across all cohorts", 
                        all_sentiments, 
                        key="cohort_sentiment"
                    )
                    
                    if selected_cohort_sentiment:
                        # Get all users across all cohorts
                        all_cohort_users = []
                        for threshold in thresholds:
                            cohort_users = processed_df[processed_df["chat_count"] >= threshold]["user_email"].tolist()
                            all_cohort_users.extend(cohort_users)
                        
                        # Remove duplicates
                        all_cohort_users = list(set(all_cohort_users))
                        
                        # Create a dataframe with all cohort questions
                        all_cohort_data = df[df["user_email"].isin(all_cohort_users)].copy()
                        
                        # Add sentiment column if not present
                        if 'sentiment' not in all_cohort_data.columns:
                            all_cohort_data['sentiment'] = all_cohort_data['question'].apply(analyze_sentiment)
                        
                        # Filter by selected sentiment
                        filtered_by_sentiment = all_cohort_data[all_cohort_data['sentiment'] == selected_cohort_sentiment]
                        
                        st.write(f"**{len(filtered_by_sentiment)} questions with '{selected_cohort_sentiment}' sentiment across all cohorts:**")
                        
                        # Display questions in a table
                        sentiment_table = filtered_by_sentiment[['user_email', 'question']].reset_index(drop=True)
                        st.dataframe(sentiment_table, use_container_width=True)
                else:
                    st.warning("No sentiment data available for cohorts.")
            
        # Additional analysis
        st.header("Detailed Cohort Breakdown")
        
        # Create cohort summary table
        cohort_summary = []
        for threshold in thresholds:
            cohort_users = processed_df[processed_df["chat_count"] >= threshold]
            
            summary = {
                "Cohort": f">= {threshold} Chats",
                "User Count": len(cohort_users),
                "% of Total Users": f"{len(cohort_users) / len(processed_df) * 100:.1f}%",
                "Avg Chats": round(cohort_users["chat_count"].mean(), 2),
                "Median Chats": round(cohort_users["chat_count"].median(), 2),
                "Min Chats": cohort_users["chat_count"].min(),
                "Max Chats": cohort_users["chat_count"].max()
            }
            cohort_summary.append(summary)
        
        cohort_summary_df = pd.DataFrame(cohort_summary)
        st.dataframe(cohort_summary_df, use_container_width=True)
        
        # Filtering section
        st.header("Explore Specific Cohort")
        selected_threshold = st.slider(
            "Select chat threshold to view users with this many chats or more:",
            min_value=1,
            max_value=int(processed_df["chat_count"].max()),
            value=5,
            step=1
        )
        
        filtered_users = processed_df[processed_df["chat_count"] >= selected_threshold]
        users_in_cohort = filtered_users["user_email"].tolist()
        
        st.metric(
            f"Users with >= {selected_threshold} chats", 
            len(filtered_users),
            f"{len(filtered_users) / len(processed_df) * 100:.1f}% of total"
        )
        
        # Create tabs for user exploration
        explore_tab1, explore_tab2, explore_tab3 = st.tabs(["Users", "Question Types", "Sentiment Analysis"])
        
        with explore_tab1:
            st.dataframe(filtered_users, use_container_width=True)
            
        with explore_tab2:
            # Get question type distribution for this cohort
            question_types_df = get_question_type_distribution(df, users_in_cohort)
            
            # Display the data
            col1, col2 = st.columns([2, 3])
            
            with col1:
                st.dataframe(question_types_df, use_container_width=True)
            
            with col2:
                # Create pie chart of question types
                fig = px.pie(
                    question_types_df,
                    values="count",
                    names="question_type",
                    title=f"Question Types for Cohort >= {selected_threshold} Chats",
                    hole=0.4
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
                
            # Add interactive question type selection
            st.markdown("#### View Questions by Type")
            st.write("Click on a question type to see all questions of that type:")
            
            # Filter cohort data with original df to include all columns
            cohort_data = df[df["user_email"].isin(users_in_cohort)].copy()
            
            # Make sure we have question_type column
            if 'question_type' not in cohort_data.columns:
                if 'Prompt' in cohort_data.columns:
                    cohort_data.loc[:, 'question_type'] = cohort_data['Prompt'].apply(extract_question_type)
                else:
                    cohort_data.loc[:, 'question_type'] = 'Unknown'
            
            # Get unique question types for selection
            question_types = cohort_data['question_type'].unique().tolist()
            
            # Create selection widget
            selected_type = st.selectbox("Select a question type", question_types, key=f"explore_type_{selected_threshold}")
            
            # Show questions of selected type
            if selected_type:
                filtered_questions = cohort_data[cohort_data['question_type'] == selected_type]
                st.write(f"**{len(filtered_questions)} questions of type '{selected_type}':**")
                
                # Display questions in a table
                questions_table = filtered_questions[['user_email', 'question']].reset_index(drop=True)
                st.dataframe(questions_table, use_container_width=True)
                
        with explore_tab3:
            # Get sentiment distribution for this cohort
            sentiment_df = get_sentiment_distribution(df, users_in_cohort)
            
            # Display sentiment analysis results
            st.markdown("#### Sentiment Analysis of Questions in This Cohort")
            
            # Display table and chart
            col1, col2 = st.columns([2, 3])
            
            with col1:
                st.dataframe(sentiment_df, use_container_width=True)
            
            with col2:
                # Create pie chart of sentiment distribution
                colors = {'Positive': '#28a745', 'Neutral': '#ffc107', 'Negative': '#dc3545'}
                fig = px.pie(
                    sentiment_df,
                    values="count",
                    names="sentiment",
                    title=f"Sentiment Distribution for Cohort >= {selected_threshold} Chats",
                    color="sentiment",
                    color_discrete_map=colors,
                    hole=0.4
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
                
            # Show questions by sentiment
            st.markdown("#### View Questions by Sentiment")
            
            # Get sentiment for all questions in this cohort
            if 'sentiment' not in cohort_data.columns:
                cohort_data['sentiment'] = cohort_data['question'].apply(analyze_sentiment)
                
            # Create selection widget for sentiment
            sentiments = cohort_data['sentiment'].unique().tolist()
            selected_sentiment = st.selectbox("Select a sentiment to view questions", sentiments, key=f"explore_sentiment_{selected_threshold}")
            
            # Show questions with selected sentiment
            if selected_sentiment:
                sentiment_questions = cohort_data[cohort_data['sentiment'] == selected_sentiment]
                st.write(f"**{len(sentiment_questions)} questions with '{selected_sentiment}' sentiment:**")
                
                # Display questions in a table
                sentiment_table = sentiment_questions[['user_email', 'question']].reset_index(drop=True)
                st.dataframe(sentiment_table, use_container_width=True)
                
except Exception as e:
    st.error(f"An error occurred while processing your data: {str(e)}")
    st.exception(e)
    
    # Example of expected data format
    st.header("Expected Data Format")
    example_data = {
        "user_email": ["user1@example.com", "user1@example.com", "user2@example.com", 
                       "user3@example.com", "user3@example.com", "user3@example.com"],
        "question": ["How do I reset my password?", "Can I change my username?", 
                     "Where is the settings menu?", "Is there a mobile app?", 
                     "How do I link my account?", "Can I export my data?"]
    }
    example_df = pd.DataFrame(example_data)
    st.dataframe(example_df)
    
    # Show what the processed data will look like
    st.header("Example of Processed Data")
    processed_example = {
        "user_email": ["user1@example.com", "user2@example.com", "user3@example.com"],
        "chat_count": [2, 1, 3]
    }
    processed_df = pd.DataFrame(processed_example)
    st.dataframe(processed_df)
    
    st.markdown("""
        Your CSV file should have at least the following columns:
        - **user_email**: The email of the user who asked the question
        - **question**: The text of the question asked
        
        The dashboard will automatically:
        1. Count how many questions each user has asked
        2. Segment users into cohorts based on question count
        3. Visualize the distribution of users across cohorts
    """)
