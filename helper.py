from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
import datetime
import matplotlib.pyplot as plt
# from matplotlib import font_manager
# font_path = r"C:\Users\karsu\Downloads\Noto_Emoji\NotoEmoji-VariableFont_wght.ttf"
# font_prop = font_manager.FontProperties(fname=font_path)
# plt.rcParams['font.family'] = font_prop.get_name()

extract = URLExtract()



def fetch_stats(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # fetch number of media messages 
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    # fetch number of links shared
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages,len(words),num_media_messages,len(links)

def most_busy_users(df):
     # Count the number of messages by each user and take the top ones
    x = df['user'].value_counts().head()
    # Calculate the percentage of messages each user sent
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    # Return the top message senders and their percentage
    return x,df

def create_wordcloud(selected_user,df):

    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user,df):

    f = open("stop_hinglish.txt",'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])  # Updated to use emoji.EMOJI_DATA

    # Convert emoji list to DataFrame and count occurrences
    emoji_df = pd.DataFrame(Counter(emojis).most_common(), columns=['Emoji', 'Count'])

    return emoji_df


def monthly_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

def daily_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap

def filter_messages_by_user(df, user):
    if user == 'Overall':
        return df
    else:
        return df[df['user'] == user]

def extract_datetime_features(df):
    # Ensure 'date' column is in string format and handle NaNs
    df['date'] = df['date'].astype(str).str.replace('\u202f', ' ', regex=True)

    # Parse the date with error handling
    try:
        df['only date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %I:%M %p', dayfirst=True).dt.date
        df['year'] = pd.to_datetime(df['date'], format='%d/%m/%Y %I:%M %p', dayfirst=True).dt.year
        df['month_num'] = pd.to_datetime(df['date'], format='%d/%m/%Y %I:%M %p', dayfirst=True).dt.month
        df['month'] = pd.to_datetime(df['date'], format='%d/%m/%Y %I:%M %p', dayfirst=True).dt.month_name()
        df['day'] = pd.to_datetime(df['date'], format='%d/%m/%Y %I:%M %p', dayfirst=True).dt.day
        df['day_name'] = pd.to_datetime(df['date'], format='%d/%m/%Y %I:%M %p', dayfirst=True).dt.day_name()
        df['hour'] = pd.to_datetime(df['date'], format='%d/%m/%Y %I:%M %p', dayfirst=True).dt.hour
        df['minute'] = pd.to_datetime(df['date'], format='%d/%m/%Y %I:%M %p', dayfirst=True).dt.minute
    except Exception as e:
        print("Error parsing dates:", e)

    return df

#SENTIMENT ANALYSIS PART

# Function to create a pie chart for sentiment distribution
def sentiment_pie_chart(selected_user, df):
    sentiment_counts = df['sentiment'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=['#66b3ff','#99ff99','#ff9999'])
    ax.set_title('Sentiment Analysis of Messages')
    return fig

# Sentiment over time
def sentiment_over_time(selected_user, df):
    sentiment_time = df.groupby(['only_date', 'sentiment']).size().unstack(fill_value=0)
    fig, ax = plt.subplots()
    sentiment_time.plot(kind='line', ax=ax)
    ax.set_title('Sentiment Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Message Count')
    return fig
