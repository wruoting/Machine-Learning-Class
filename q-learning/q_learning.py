import tensorflow as tf
from csv_to_dataframe import csv_to_dataframe
from graph import graph


def q_learning():
    df_bullish = csv_to_dataframe('bullish-test.csv')
    df_test = csv_to_dataframe('overlay-test.csv')
    graph([df_bullish,df_test],'Plots.html')

q_learning()
