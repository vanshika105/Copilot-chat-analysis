import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Get database URL from environment
DATABASE_URL = os.environ.get("DATABASE_URL")

# Create SQLAlchemy engine and session
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
Base = declarative_base()

# Define models
class UserQuestion(Base):
    __tablename__ = "user_questions"
    
    id = Column(Integer, primary_key=True)
    user_email = Column(String(255), nullable=False)
    question = Column(Text, nullable=False)
    created_at = Column(DateTime, default=func.now())
    
    def __repr__(self):
        return f"<UserQuestion(user_email='{self.user_email}', question='{self.question[:20]}...')>"

# Functions to interact with the database
def save_data_to_db(data_df):
    """
    Save data from a pandas DataFrame to the database.
    
    Args:
        data_df (pd.DataFrame): DataFrame with user_email and question columns
    
    Returns:
        int: Number of records saved
    """
    session = Session()
    count = 0
    
    try:
        # Convert DataFrame to list of dictionaries
        records = data_df.to_dict('records')
        
        # Insert records into the database
        for record in records:
            user_question = UserQuestion(
                user_email=record['user_email'],
                question=record['question']
            )
            session.add(user_question)
            count += 1
        
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()
    
    return count

def load_data_from_db():
    """
    Load all user question data from the database.
    
    Returns:
        pd.DataFrame: DataFrame containing all user questions
    """
    session = Session()
    
    try:
        # Query all records
        query = session.query(
            UserQuestion.user_email,
            UserQuestion.question,
            UserQuestion.created_at
        )
        
        # Convert to DataFrame
        df = pd.read_sql(query.statement, session.bind)
        
    except Exception as e:
        raise e
    finally:
        session.close()
    
    return df

def get_user_count():
    """
    Get the count of unique users in the database.
    
    Returns:
        int: Number of unique users
    """
    session = Session()
    
    try:
        count = session.query(func.count(func.distinct(UserQuestion.user_email))).scalar()
    except Exception as e:
        raise e
    finally:
        session.close()
    
    return count

def get_question_count():
    """
    Get the total count of questions in the database.
    
    Returns:
        int: Number of questions
    """
    session = Session()
    
    try:
        count = session.query(func.count(UserQuestion.id)).scalar()
    except Exception as e:
        raise e
    finally:
        session.close()
    
    return count