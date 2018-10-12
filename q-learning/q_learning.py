import tensorflow as tf
from csv_to_dataframe import csv_to_dataframe
from graph import graph
import numpy as np
import pandas as pd
from preprocessing import processing_data
from q_model import q_model
import random


def q_learning():
    df_bullish = csv_to_dataframe('bullish-test.csv')
    df_test = csv_to_dataframe('overlay-test.csv')
    model = q_model()
    state, data = processing_data(df_bullish) # typing in: dataframe

    total_state = pd.Series(index=np.arange(len(data))) # the length of the data
    stored_buffer = 400
    batch_size = 15
    gamma = 0.9 # discount factor, we are rewarding long term trading with this high factor
    status = 1
    terminal_state = 0
    time_step = 1
    while status == 1:
        # We start in state S
        # Run the Q function on S and get the predicted value on Q
        q_value = model.predict(state, batch_size=1)

        # Softmax it
        q_value_list, softmax_percentages = softmax(q_value)
        choice = np.random.choice(3, 1, p=softmax_percentages)
        action = q_value_list[choice]

        # take action and return new state
        new_state, time_step, total_state, terminal_state = take_action(data, action, total_state, time_step)
        # observe reward
        reward = get_reward(new_state, time_step, action, data, total_state, terminal_state)

        # Store states if less than our buffer
        # Train on a random subset of training sets in our buffer
        if len(replay) < stored_buffer:
            replay.append((state, action, reward, new_state))
        else:
            replay[0] = (state, action, reward, new_state)
            mini_batch = random.sample(replay, batch_size)
            x_train = []
            y_train = []
            for batch in mini_batch:
                # Get max_Q(S',a)
                old_state, action, reward, new_state = batch
                old_q_val = model.predict(old_state, batch_size=1)
                new_q = model.predict(new_state, batch_size=1)
                max_q = np.max(new_q)
                y = np.zeros((1, 3))
                y[:] = old_q_val[:]
                if terminal_state == 0:  # non-terminal state
                    update = (reward + (gamma * max_q))
                else:  # terminal state
                    update = reward
                y[0][action] = update
                # print(time_step, reward, terminal_state)
                x_train.append(old_state)
                y_train.append(y.reshape(3, ))

            x_train = np.squeeze(np.array(x_train), axis=1)
            y_train = np.array(y_train)
            model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=1, verbose=0)
            state = new_state
        if terminal_state == 1:  # if reached terminal state, update epoch status
            status = 0

    eval_reward = evaluate_Q(test_data, model, price_data, i)
    learning_progress.append(eval_reward)


    # Init all states and actions
    # graph([df_bullish,df_test],'Plots.html')


# we're not going to use an anneal value because this is just a proportion of q values
def softmax(q_values):
    q_value_softmax = []
    q_value_list = []
    sum_exponent = np.exp(np.sum(q_values))
    for value in q_values:
        q_value_softmax.append(np.exp(value) / sum_exponent)
        q_value_list.append(value)
    return q_value_list, q_value_softmax


def take_action(data, action, total_state, time_step):
    time_step += 1
    terminal_state = 0
    # if the current state is the last point in the frame
    state = data[time_step - 1:time_step, 0:1, :]
    if time_step + 1 == data.shape[0]:
        terminal_state = 1
        total_state.loc[time_step] = 0

        return state, time_step, total_state, terminal_state
    state = data[time_step-1:time_step]
    #take action
    if action == 1: #buy
        total_state.loc[time_step] = 1
    elif action == 2: #sell
        total_state.loc[time_step] = -1
    else: #hold
        total_state.loc[time_step] = 0

    return state, time_step, total_state, terminal_state


q_learning()
