import tensorflow as tf
from csv_to_dataframe import csv_to_dataframe
from keras.models import load_model
from q_learning import take_action, get_reward
from graph import graph, map_to_graph
from preprocessing import processing_data
import numpy as np
import pandas as pd
import random

def q_evaluate():
    # reset time step to evaluate the total reward
    df_bullish = csv_to_dataframe('btc-test.csv')
    plot_name = 'BTC-Test.html'
    model = load_model('q_model_btc_train.h5')
    stored_buffer = 50
    batch_range = range(0, 15)
    state, price_data, unscaled_price_data = processing_data(df_bullish, batch_range)
    decision_state = pd.Series(index=np.arange(len(price_data)))
    eval_reward, decision_state = evaluate_q_epoch(state, price_data, unscaled_price_data, model, decision_state, batch_range, stored_buffer)
    print("Reward:" + str(eval_reward))
    # Init all states and actions
    decision_state_dataframe = pd.DataFrame(np.transpose([df_bullish['Time'],decision_state]))

    color_set = ['rgb(0,128,0)','rgb(255,0,0)']
    buy_to_map, sell_to_map = map_to_graph(decision_state_dataframe, df_bullish, color_set)
    graph([df_bullish], [buy_to_map], [sell_to_map], plot_name)


def evaluate_q_epoch(state, price_data, unscaled_price_data, model, decision_state, batch_range, stored_buffer):
    eval_reward = 0
    terminal_state = False
    state_index = 0
    max_length = len(state)
    batch_size = len(batch_range)
    while not terminal_state:
        reshaped_data = np.reshape(state[state_index],(1,1,np.multiply(len(batch_range),2)))
        q_value = model.predict(reshaped_data, batch_size=batch_size)
        q_value = q_value[0][0]
        action = np.argmax(q_value)
        state_index, decision_state, terminal_state = take_action(action, decision_state, state, state_index, stored_buffer, batch_range, eval=True)
        action_to_reward = get_reward(state_index, action, price_data, unscaled_price_data, decision_state, batch_size, eval=True)
        eval_reward += action_to_reward
        if state_index < max_length - 1:
            state_index += 1
        else:
            terminal_state = True
    return eval_reward, decision_state
q_evaluate()
