from urllib.request import urlopen, Request
from urllib.error import HTTPError
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime, timedelta
from suggestions import fetch_news_data, fetch_stock_price_data, combine_sentiment_with_price_data, enhanced_buy_sell_recommendation

nltk.download('vader_lexicon')

def clear_frame(frame):
    for widget in frame.winfo_children():
        widget.destroy()

def display_graph(selected_tickers):
    clear_frame(graph_frame)
    clear_frame(inner_frame)
    
    all_data = pd.DataFrame()
    recommendations = []
    for ticker in selected_tickers:
        sentiment_df = fetch_news_data(ticker)
        if sentiment_df.empty:
            print(f"No data found for {ticker}")
            continue
        vader = SentimentIntensityAnalyzer()
        sentiment_df['compound'] = sentiment_df['title'].apply(lambda title: vader.polarity_scores(title)['compound'])
        sentiment_df['date'] = sentiment_df['date'].str.strip()
        sentiment_df = sentiment_df[sentiment_df['date'].notnull() & (sentiment_df['date'] != '')]

        try:
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date'], errors='coerce').dt.date
            sentiment_df = sentiment_df[sentiment_df['date'].notnull()]
        except Exception as e:
            print(f"Error parsing dates for {ticker}: {e}")
            continue

        if sentiment_df.empty:
            print(f"No valid date data for {ticker}")
            continue

        price_df = fetch_stock_price_data(ticker)
        combined_df = combine_sentiment_with_price_data(sentiment_df, price_df)
        combined_df = enhanced_buy_sell_recommendation(combined_df)

        combined_df['ticker'] = ticker
        all_data = pd.concat([all_data, combined_df])
        
        # Gather recommendations
        ticker_recommendations = combined_df[['date', 'recommendation']].drop_duplicates()
        for _, row in ticker_recommendations.iterrows():
            recommendations.append(f"{ticker} on {row['date']}: {row['recommendation']}")

    if all_data.empty:
        print("No data available for any ticker in the past few days")
        return

    # Display recommendations in the inner_frame
    recommendation_text = "\n".join(recommendations)
    recommendation_label = tk.Label(inner_frame, text=recommendation_text, justify=tk.LEFT)
    recommendation_label.pack()

    plt.figure(figsize=(12, 8))
    colors = ['g', 'b', 'r', 'c', 'm', 'y', 'k']
    color_dict = {ticker: colors[i % len(colors)] for i, ticker in enumerate(selected_tickers)}

    for ticker in selected_tickers:
        ticker_data = all_data[all_data['ticker'] == ticker]
        mean_df = ticker_data.groupby(['date'])['compound'].mean()
        mean_df.index = pd.to_datetime(mean_df.index)  # Ensure the index is datetime
        mean_df.plot(kind='bar', color=color_dict[ticker], label=ticker, alpha=0.5)

    plt.title('Sentiment Analysis for Selected Stocks (Past few days)')
    plt.ylabel('Compound Sentiment Score')
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.legend()

    fig = plt.gcf()
    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

root = tk.Tk()
root.title("Stock Sentiment Analysis")

dropdown_label = tk.Label(root, text="Select Stocks (Ctrl+Click to select multiple):")
dropdown_label.pack()

listbox = tk.Listbox(root, selectmode=tk.MULTIPLE, height=10)
tickers = ['AMZN', 'GOOG', 'FB', 'AAPL', 'MSFT', 'TSLA', 'NVDA', 'NFLX', 'ADBE', 'INTC']
for ticker in tickers:
    listbox.insert(tk.END, ticker)
listbox.pack()

button = tk.Button(root, text="Show Graph", command=lambda: display_graph([listbox.get(i) for i in listbox.curselection()]))
button.pack()

# Create frames for graph and recommendations
graph_frame = tk.Frame(root)
graph_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

recommendation_frame = tk.Frame(root)
recommendation_frame.pack(side=tk.RIGHT, fill=tk.Y)

# Create a canvas with a scrollbar for the recommendations
canvas = tk.Canvas(recommendation_frame)
scrollbar = tk.Scrollbar(recommendation_frame, orient="vertical", command=canvas.yview)
canvas.configure(yscrollcommand=scrollbar.set)

scrollbar.pack(side="right", fill="y")
canvas.pack(side="left", fill="both", expand=True)

inner_frame = tk.Frame(canvas)
canvas.create_window((0, 0), window=inner_frame, anchor="nw")

inner_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

root.mainloop()

