import re
import sklearn
import streamlit as st
import pandas as pd
from datetime import datetime
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import joblib
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np

nltk.download('vader_lexicon')
nltk.download('stopwords')


def preprocess(data):
    # Updated pattern to handle both 12-hour and 24-hour time formats
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\u202f?[APMapm]{2}\s-\s'
    
    # Split messages and extract dates
    messages = re.split(pattern, data)[1:]  # Skip the first split which might be empty
    dates = re.findall(pattern, data)

    # Create DataFrame
    dataf = pd.DataFrame({'user_message': messages, 'message_date': dates})
    # #Behaviour Model 
    

    # Handle both 12-hour (AM/PM) and 24-hour formats
    try:
        # First attempt to parse as 12-hour format (with AM/PM)
        dataf['message_date'] = pd.to_datetime(dataf['message_date'], format='%d/%m/%Y, %I:%M %p - ')
    except ValueError:
        # If it fails, fall back to parsing as 24-hour format
        try:
            dataf['message_date'] = pd.to_datetime(dataf['message_date'], format='%d/%m/%Y, %H:%M - ')
        except ValueError:
            # Fallback for non-breaking space parsing
            dataf['message_date'] = pd.to_datetime(dataf['message_date'], format='%d/%m/%Y, %H:%M\u202f - ')

    # Rename column for clarity
    dataf.rename(columns={'message_date': 'date'}, inplace=True)

    users = []
    messages = []
    
    for message in dataf['user_message']: 
        # Handle user split logic (improved for consistency)
        entry = re.split(r'([\w\W]+?):\s', message)
        if len(entry) > 1:  # If the split finds a user
            users.append(entry[1])
            messages.append(entry[2])  # Collect the message after user
        else:
            users.append('group_notification')
            messages.append(entry[0])  # It's a notification if no user is found

    # Add users and messages to DataFrame
    dataf['user'] = users
    dataf['message'] = messages

    # Drop the original user_message column
    dataf.drop(columns=['user_message'], inplace=True)
    dataf = dataf[dataf['user'] != 'group_notification']
    
    # Add sentiment analysis to the DataFrame when you preprocess the data
    dataf['sentiment'] = dataf['message'].apply(get_sentiment_vader)

    # Extract additional date and time-related columns
    dataf['only_date'] = dataf['date'].dt.date
    dataf['year'] = dataf['date'].dt.year
    dataf['month_num'] = dataf['date'].dt.month
    dataf['month'] = dataf['date'].dt.month_name()
    dataf['day'] = dataf['date'].dt.day
    dataf['day_name'] = dataf['date'].dt.day_name()
    dataf['hour'] = dataf['date'].dt.hour
    dataf['minute'] = dataf['date'].dt.minute
    dataf['day_of_week'] = dataf['date'].dt.day_of_week
    dataf['is_weekend'] = dataf['day_of_week'].apply(lambda x: x >= 5)
    dataf['message_length'] = dataf['message'].apply(len)
    # Create 'daily_message_intensity' by multiplying 'hour' with 'message_length'
    dataf.loc[:, 'daily_message_intensity'] = dataf['day'] * dataf['message_length']
    # Create 'day_weekend_interaction' by multiplying 'day_of_week' with 'is_weekend'
    dataf.loc[:, 'day_weekend_interaction'] = dataf['day_of_week'] * dataf['is_weekend']

    # Calculate the period for each message
    dataf['period'] = dataf['hour'].apply(lambda x: f'{x:02d}-{(x+1)%24:02d}')
    # dataf['sentiment_score'] = dataf['sentiment'].apply(lambda x: x['compound'])
    
    return dataf


#FREQUENCY PREDICTER  ------   How many messages they will send in a given time frame (e.g., per hour or day).
def prepare_data(dataf):    
    print("DataFrame shape before merging:", dataf.shape)
    # Group by user and hour to get the number of messages sent
    user_day_count = dataf.groupby(['user', 'day']).size().reset_index(name='message_count')
    # Group by total number of messages
    total_messages = dataf.groupby('user').size().reset_index(name='total_messages')
    # Group by date and count messages for all users
    daily_messages = dataf.groupby(['user', 'only_date']).size().reset_index(name='total_daily_messages')
    
    # Merge with the original DataFrame to retain additional features
    dataf = dataf.merge(user_day_count, on=['user', 'day'], how='left')
    print("After merging user_day_count, columns:", dataf.columns)
    dataf = dataf.merge(total_messages, on='user', how='left')
    print("After merging total_mesxsages, columns:", dataf.columns)
    dataf = dataf.merge(daily_messages, on=['user', 'only_date'], how='left')
    print("After merging daily_messages, columns:", dataf.columns)

    print("DataFrame shape after merging:", dataf.shape)
    print("Columns after preparation:", dataf.columns)
    
    return dataf

def train_datas(dataf, selected_user):
    dataf = prepare_data(dataf)
    print("DataFrame columns before feature selection:", dataf.columns)
    # Overall_df = prepare_data(dataf)
    
    if selected_user == "Overall":
        # Use overall data; ensure total_messages is included
        print("Using overall data for training.")
        # dataf = Overall_df
        if selected_user == "Overall":
            print(dataf.head())

        if 'total_messages' not in dataf.columns:
            print("Error: 'total_messages' column is missing from DataFrame.")
            return None
        

    else:
        dataf = dataf[dataf['user'] == selected_user]
        print(f"Filtered data for {selected_user}: {dataf.shape}")

        if dataf.empty:
            print("No data available for the selected user.")
            return None
    
    print(f"Data columns after filtering by user: {dataf.columns}")
    print(dataf.head())
    
    # Define features and target variable
    X = dataf[['minute', 'hour', 'day', 'day_of_week', 'is_weekend', 'message_length','daily_message_intensity', 'day_weekend_interaction', 'total_messages', 'total_daily_messages']]
    y = dataf['message_count'] #if selected_user != "Overall" else dataf['total_daily_messages'] 
    
    # print("DATAFRAME AFTER FEATURES SELECTION:", dataf.columns)
    
    if X.empty or y.empty:
        print("X or y is empty, cannot train the model.")
        return None
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print("R-squared:", f"{r2:.2f}")
    print("Mean Squared Error:", f"{mse:.2f}")
    
    st.write(f"R-squared: {r2}")
    st.write(f"Mean Squared Error: {mse}")
    
    predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': f"{y_pred}"})
    print(predictions_df.head())
        
    joblib.dump(model, 'linear_regression_model.pkl')    

def load_model():
    return joblib.load('linear_regression_model.pkl')


# Initialize VADER sentiment intensity analyzer
sid = SentimentIntensityAnalyzer()

#2. Add Sentiment Analysis to the preprocess Function
def get_sentiment_vader(message):
    sentiment = sid.polarity_scores(message)
    if sentiment['compound'] >= 0.05:
        return 'Positive'
    elif sentiment['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'
