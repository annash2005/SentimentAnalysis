import logging
import os
import pickle
from urllib.request import urlopen, Request
from urllib.error import HTTPError
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize logging
logging.basicConfig(filename='app.log', level=logging.ERROR)

CACHE_DIR = 'cache'

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def fetch_news_data(ticker):
    finviz_url = 'https://finviz.com/quote.ashx?t='
    url = finviz_url + ticker

    cache_path = os.path.join(CACHE_DIR, f"{ticker}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    try:
        req = Request(url=url, headers={'user-agent': 'my-app'})
        response = urlopen(req)
    except HTTPError as e:
        logging.error(f"HTTP Error for {ticker}: {e.code}")
        print(f"HTTP Error for {ticker}: {e.code}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

    html = BeautifulSoup(response, features='html.parser')
    news_table = html.find(id='news-table')

    parsed_data = []
    
    if news_table:
        for row in news_table.findAll('tr'):
            a_tag = row.find('a')
            if a_tag is None:
                continue
            
            title = a_tag.text
            date_data = row.td.text.strip().split(' ')
            
            if len(date_data) == 1:
                time = date_data[0]
                date = None
            else:
                date = date_data[0]
                time = date_data[1]
            if date is None or time is None:
                continue
            
            parsed_data.append([ticker, date, time, title])
    
    df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])
    with open(cache_path, 'wb') as f:
        pickle.dump(df, f)

    return df

def calculate_moving_average(data, window):
    return data.rolling(window=window).mean()

def fetch_stock_price_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1mo")  # Fetch last month's data
    hist['MA5'] = calculate_moving_average(hist['Close'], 5)
    hist['MA20'] = calculate_moving_average(hist['Close'], 20)
    return hist

def combine_sentiment_with_price_data(sentiment_df, price_df):
    combined_df = sentiment_df.copy()
    combined_df['date'] = pd.to_datetime(combined_df['date'])
    combined_df['date'] = combined_df['date'].dt.tz_localize(None)  # Make naive (remove timezone)
    
    price_df.reset_index(inplace=True)  # Ensure Date is not the index
    price_df['Date'] = pd.to_datetime(price_df['Date'])
    price_df['Date'] = price_df['Date'].dt.tz_localize(None)  # Make naive (remove timezone)
    
    combined_df = pd.merge(combined_df, price_df, left_on='date', right_on='Date', how='left')
    
    combined_df['MA5'] = combined_df['MA5'].fillna(method='ffill')
    combined_df['MA20'] = combined_df['MA20'].fillna(method='ffill')
    
    return combined_df

def enhanced_buy_sell_recommendation(df):
    recommendation = []
    for index, row in df.iterrows():
        sentiment = row['compound']
        ma5 = row['MA5']
        ma20 = row['MA20']
        
        if sentiment > 0.2 and ma5 > ma20:
            recommendation.append("Buy")
        elif sentiment < -0.2 and ma5 < ma20:
            recommendation.append("Sell")
        else:
            recommendation.append("Hold")
    df['recommendation'] = recommendation
    return df
