import tensorflow as tf
from csv_to_dataframe import csv_to_dataframe
from keras.models import load_model
from q_learning import take_action, evaluate_q_epoch
from graph import graph, map_to_graph
from preprocessing import processing_data
import numpy as np
import pandas as pd
import random

def q_evaluate():
    # reset time step to evaluate the total reward
    df_bullish = csv_to_dataframe('sin-test.csv')
    plot_name = 'Sine.html'
    model = load_model('q_model.h5')
    stored_buffer = 50
    batch_range = range(0,15)
    state, price_data = processing_data(df_bullish,batch_range)
    decision_state = pd.Series(index=np.arange(len(price_data)))
    eval_reward, decision_state = evaluate_q_epoch(state, price_data, model, decision_state, batch_range, stored_buffer)
    print("Reward:" + str(eval_reward))
    # Init all states and actions
    decision_state_dataframe = pd.DataFrame(np.transpose([df_bullish['Time'],decision_state]))

    color_set = ['rgb(0,128,0)','rgb(255,0,0)']
    buy_to_map, sell_to_map = map_to_graph(decision_state_dataframe, df_bullish, color_set)
    graph([df_bullish], [buy_to_map], [sell_to_map], plot_name)

q_evaluate();
