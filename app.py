import streamlit as st
import preprocessor
from preprocessor import prepare_data, load_model, train_datas
import helper
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os 
import time
import atexit

#REAL TIME

file_path = r"C:\Users\karsu\OneDrive\Desktop\Programming\venv\CollegeProject\WhatsApp Chat with Doubt Grp 11th PLT B5 G4.txt"
global df
class ChatFileHandler(FileSystemEventHandler):
    def __init__(self, file_path, callback):
        self.file_path = file_path
        self.callback = callback
            
    def on_modified(self, event):
        if(event.src_path == self.file_path):
            print(f"File modified: {event.src_path}")  # Debugging line
            self.callback()             # Call the function to update the data

def start_file_watcher(file_path, callback):
    event_handler = ChatFileHandler(file_path, callback)
    observer = Observer()
    observer.schedule(event_handler, os.path.dirname(file_path), recursive=False)
    observer.start()
    return observer

def update_chat_data():
    global df
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.read()
        df = preprocessor.preprocess(data)
        st.experimental_rerun()  # refresh the app
    except Exception as e:
        st.error(f"Error reading the chat file: {e}")

def load_chat_data():
    global df   
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.read()
        df = preprocessor.preprocess(data)
    except Exception as e:
        st.error(f"Error reading the chat file: {e}")

st.title("WhatsApp Chat Analyzer")

# Sidebar for file upload
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    
    file_path = "./temp_chat.txt"           #Temporary file for saving chat data file.
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getvalue())
    
    #Initialize the dataset
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    df = preprocessor.preprocess(data)
    
    # Fetch unique users
    user_list = df['user'].unique().tolist()
    user_list.sort()
    user_list.insert(0, "Overall")
    
    # print("User_list: ", user_list)
    
    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    if st.sidebar.button("Show Analysis"):
        # Stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)

        # Monthly Timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Daily Timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)
        
        # Sentiment Analysis
        st.title("Sentiment Analysis")                                                      ####### Check this out.
        st.pyplot(helper.sentiment_pie_chart(selected_user, df))
        st.pyplot(helper.sentiment_over_time(selected_user, df))

        # Activity Maps
        st.title('Activity Map')
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most Busy Day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most Busy Month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Weekly Activity Map
        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        # Most Busy Users (Group Level)
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x, new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)
            with col1:
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)# WordCloud
        

        # Most Common Words
        most_common_df = helper.most_common_words(selected_user, df)
        fig, ax = plt.subplots()
        ax.barh(most_common_df[0], most_common_df[1])
        plt.xticks(rotation='vertical')
        st.title('Most Common Words')
        st.pyplot(fig)

        # Emoji Analysis
        emoji_df = helper.emoji_helper(selected_user, df)
        st.title("Emoji Analysis")
        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig, ax = plt.subplots()
            ax.pie(emoji_df['Count'], labels=emoji_df['Emoji'], autopct="%0.2f%%")
            st.pyplot(fig)

        #FREQUENCY PREDICTER    
        # user_df = df[df['user'] == selected_user]
        new_features = prepare_data(df)
        train_datas(df, selected_user)
        print(f"Selected user: {selected_user}")
        print("DataFrame passed to train_datas shape:", df.shape)
        print("DataFrame passed to train_datas columns:", df.columns)
        
        models = load_model()
        
        #Later, to make predictions on new data:
        
        # new_features['hoursq'] = new_features['hour'] * new_features['hour']
        # print(f"Data after preparation: {new_features.shape}")
        X_new = new_features[['minute', 'hour', 'day', 'day_of_week', 'is_weekend', 'message_length', 'daily_message_intensity', 'day_weekend_interaction', 'total_messages', 'total_daily_messages']]
        print("DataFrame passed to model before prediction (Overall):", X_new.shape)
        print(df.head())
        if X_new.shape[0] == 0:
            st.error("No data available for prediction!")
        
        predictions = models.predict(X_new)   

        print("Rows after passing to model")
        print(df.head())
        # st.write("The DataFrame is : ")
        # st.dataframe(df)
        predictions_df = pd.DataFrame({
        'Minute': X_new['minute'],
        'Hour': X_new['hour'],
        'Day': X_new['day'],
        'Day of Week': X_new['day_of_week'],
        'Is Weekend': X_new['is_weekend'],
        'Message Length': X_new['message_length'],
        'Daily Message Intensity': X_new['daily_message_intensity'],
        'Day Weekend Interaction': X_new['day_weekend_interaction'],
        'Total Messages': X_new['total_messages'],
        'Total Message Intensity': X_new['total_daily_messages'] ,
        'Predictions': predictions
        # 'Sentiment': X_new['sentiment']
        })
        predictions_df['user'] = df['user']
        
        #DISPLAY PREDICTIONS IN STREAMLIT
        st.title("Message Count Predictions")
        st.write("Predictions DataFrame:")
        st.dataframe(predictions_df)
        
        
        load_chat_data()
        if __name__ == '__main__':
            # Start the file watcher
            observer = start_file_watcher(file_path, update_chat_data)
            # Ensuring oberserver is stopped whenever the app exits.
            import atexit
            atexit.register(observer.stop)
            
            if 'df' in globals() and df is not None:
                st.write("Main DataFrame: ")
                st.dataframe(df)  # Display the DataFrame if it's loaded
            else:
                st.warning("No data to display yet. Modify the file to load updates.")
        
        
        # FILTERED MESSAGES
        filtered_df = helper.filter_messages_by_user(df, selected_user)
        # st.dataframe(filtered_df[['user', 'message', 'date']])R

        # Sentiment Analysis for Selected User
        sentiment_counts = filtered_df['sentiment'].value_counts()
        st.write(f"Sentiment Analysis for {selected_user}")
        fig, ax = plt.subplots()
        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

               

        
  
        
        
        # # Extract DateTime Features
        # df = helper.extract_datetime_features(df)
        # st.dataframe(df.head(50))

        # Date Filters
        # year_filter = st.sidebar.selectbox('Select Year', df['year'].dropna().unique())
        # month_filter = st.sidebar.selectbox('Select Month', df['month'].dropna().unique())
        # date_filter = st.sidebar.selectbox('Select Date', df['date'].dropna().unique())
 
        # filtereddate_df = df[(df['year'] == year_filter) & (df['month'] == month_filter) & (df['date'] == date_filter)]
        # st.dataframe(filtereddate_df)  # Uncomment after testing

# if "df" not in st.session_state:
#     st.session_state.df = df        # Access st.session_state.df to use the processed data
