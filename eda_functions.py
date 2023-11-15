import pandas as pd
import numpy as np
import geocoder
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

def get_and_clean_the_data(path):

    df = pd.read_csv(path)
    
    # Drop null rows
    df = df.dropna()

    # Create xaxis variable
    df['year'] = df['created_at'].apply(lambda x: x.split('-')[0])

    # Get number of data each year
    data_count_by_year = df['year'].value_counts().sort_index()
    target_count = data_count_by_year.iloc[5] # Choose target_count as the number of tweets in year 2011.

    # Create dataframe to store our undersampled data
    undersampled_data = pd.DataFrame()

    # Add all years. 
    for year in range(2013, 2020):
        year_data = df[df['year'] == str(year)]
        
        if len(year_data) > target_count:
            year_data = year_data.sample(n=target_count, random_state=42)  # Use a fixed random_state for reproducibility
        
        undersampled_data = pd.concat([undersampled_data, year_data])

    df = undersampled_data
    print("shape of our final dataframe:", undersampled_data.shape)

    return df

def see_words_in_column(df, column):
    return df[column].value_counts()

def count_ratings_based_on_column(df, xaxis):
    plt.figure()
    sns.countplot(df[xaxis])
    plt.title(f'Number of ratings for each {xaxis}')
    plt.show()

def pie_chart_based_on_columns(df, pie, slices):
    
    # Get the breakdown of gender for each stance
    df_plot = df.groupby([slices,pie]).size().reset_index().pivot(columns=pie, index=slices, values=0)

    # Display the plot
    df_plot.plot.pie(subplots=True, figsize = (15,15))

def bar_chart_based_on_columns(df, xaxis, yaxis):

    # Get the breakdown of aggresiveness for each stance
    df_plot = pd.crosstab(index=df[xaxis],
                        columns=df[yaxis], 
                        normalize="index")

    # Display the plot
    df_plot.plot(kind='bar',stacked=True)

    # Just for showing the percentage breakdown
    for n, x in enumerate([*df_plot.index.values]):
        for proportion in df_plot.loc[x]:
                    
            plt.text(x=n,
                    y=proportion,
                    s=f'{np.round(proportion * 100, 1)}%', 
                    color="black",
                    fontsize=12,
                    fontweight="bold")

    plt.show()

def line_chart_based_on_columns(df, xaxis, yaxis):

    # Get the average sentiment each year
    average_by_year = df.groupby(xaxis)[yaxis].mean()

    # Display the plot
    plt.figure(figsize=(10, 6))
    plt.plot(average_by_year.index, average_by_year.values, marker='o', linestyle='-')
    plt.title(f'Average {yaxis} Over the {xaxis}')
    plt.xlabel(xaxis)
    plt.ylabel(f'Average {yaxis}')
    plt.grid(True)
    plt.show()

def get_and_clean_the_data_2(path):

    tweets = pd.read_csv(path)
    tweets.dropna(inplace=True)
    tweets.head()

    # Feature engineering

    # Define function to get the date
    def get_tweet_timestamp(tid):
        offset = 1288834974657
        tstamp = (tid >> 22) + offset
        utcdttime = datetime.utcfromtimestamp(tstamp/1000)
        return utcdttime
        # print(str(tid) + " : " + str(tstamp) + " => " + str(utcdttime))

    # Create 'date' column
    tweets.date = tweets.tweetid.apply(lambda x: get_tweet_timestamp(x))
    # Create 'year' column
    tweets.year = tweets.date.dt.year
    # Create 'month' column
    tweets.month = tweets.date.dt.year

    # For simplicity, tweets labelled 2 will be replaced with 1 for sentiment as they both represent positive sentiment
    tweets.sentiment.replace(2,1,inplace=True)
    return tweets
