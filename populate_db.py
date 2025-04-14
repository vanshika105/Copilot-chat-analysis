import pandas as pd
import os
from database import UserQuestion, Base, engine, save_data_to_db

def populate_database():
    """
    Populate the database with data from the CSV file
    """
    # Create all tables if they don't exist
    Base.metadata.create_all(engine)
    
    try:
        # Read the CSV file
        df = pd.read_csv('attached_assets/query_result_2025-04-07T13_03_27.309441Z.csv')
        
        # Rename columns to match our schema
        df = df.rename(columns={
            'Userid': 'user_email',
            'Question': 'question'
        })
        
        # Save to database
        records_saved = save_data_to_db(df)
        print(f"Successfully saved {records_saved} records to the database")
        
        return True
    
    except Exception as e:
        print(f"Error populating database: {str(e)}")
        return False

if __name__ == "__main__":
    populate_database()