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
    state, price_data = processing_data(df_bullish)
        # Output:[[ Price Time ]]
        # State: [[ 1 2 ]]
        # Data: [[ 1 2 ]]
        #       [[ 3 4 ]]
        #       ...

    total_state = pd.Series(index=np.arange(len(price_data))) # the length of the data
        # Output: 0     NaN
        # 1     NaN
        # 2     NaN
        # 3     NaN

    stored_buffer = 50
    batch_size = 15
    epochs = 100
    gamma = 0.9  # discount factor, we are rewarding long term trading with this high factor
    status = 1
    terminal_state = False
    time_step = 13
    learning_progress = []
    replay = []

    for epoch in range(epochs):
        while status == 1:
            # We start in state S
            # Run the Q function on S and get the predicted value on Q
            # q_value = model.predict(state, batch_size=1)
            q_value = [0.1, 0.2, 0.7]
            # Softmax it
            q_value_list, softmax_percentages = softmax(q_value, time_step, len(price_data))
            action = int(np.random.choice(3, 1, p=softmax_percentages))

            # take action and return new state
            new_state, time_step, total_state, terminal_state = take_action(price_data, action, total_state, time_step)

            # observe reward
            reward = get_reward(new_state, time_step, action, price_data, total_state, terminal_state)

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
                    #old_q_val = model.predict(old_state, batch_size=1)
                    old_q_val = [0.1, 0.2, 0.7]
                    #new_q_val = model.predict(new_state, batch_size=1)
                    new_q_val = [0.6, 0.3, 0.1]
                    max_q = np.max(new_q_val)
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
                #model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=1, verbose=0)
                state = new_state
            if terminal_state == True:  # if reached terminal state, update epoch status
                status = 0

    eval_reward = evaluate_q_epoch(price_data, model, epochs)
    learning_progress.append(price_data)


    # Init all states and actions
    # graph([df_bullish,df_test],'Plots.html')


# we're not going to use an anneal value because this is just a proportion of q values
# an annealing value should be used closer to the end of a sequence, we can inject state here
def softmax(q_values, time_step, total_steps):

    q_value_softmax = []
    q_value_list = []
    if time_step == 0:
        annealing_factor = 0.001 # very small factor for now
    else:
        annealing_factor = time_step/total_steps
    sum_exponent = np.sum(np.exp(np.divide(q_values, annealing_factor)))
    for value in q_values:
        numerator = np.exp(np.divide(value, annealing_factor))
        q_value_softmax.append(np.divide(numerator, sum_exponent))
        q_value_list.append(value)
    return q_value_list, q_value_softmax


def take_action(data, action, total_state, time_step):
    time_step += 1
    terminal_state = False
    # if the current state is the last point in the frame
    state = data[time_step - 14:time_step, time_step - 14:time_step]

    if time_step + 1 == data.shape[0]:
        terminal_state = True
        total_state.loc[time_step] = 0

        return state, time_step, total_state, terminal_state
    state = data[time_step-14:time_step]
    #take action
    if action == 1: #buy
        total_state.loc[time_step] = 1
    elif action == 2: #sell
        total_state.loc[time_step] = -1
    else: #hold
        total_state.loc[time_step] = 0
    # jump forward 15 points
    time_step+=15
    return state, time_step, total_state, terminal_state

def get_reward(new_state, time_step, action, data, total_state, terminal_state, epoch=0):
    total_state.fillna(value=0,inplace=True)
    reward = 0
    buy_reward = data[time_step] - data[time_step-1]
    hold_reward = 0
    sell_reward = data[time_step-1] - data[time_step]
    action_to_reward = {
        0:hold_reward,
        1:buy_reward,
        2:sell_reward
    }

    return action_to_reward[reward]

def evaluate_q_epoch(price_data, model, epoch=0):
    total_state = pd.Series(index=np.arange(len(price_data)))
    eval_reward = 0
    status = 1
    terminal_state = False
    state = price_data[0]
    time_step = 1
    while status == 1:
        #q_value = model.predict(state, batch_size=1)
        q_value = [0.5, 0.3, 0.2]
        action = q_value[np.argmax(q_value)]
        new_state, time_step, total_state, terminal_state = take_action(price_data, action, total_state, time_step)
        eval_reward += get_reward(new_state, time_step, action, price_data, total_state, terminal_state, epoch=epoch)
        state = new_state
        if terminal_state == True:
            status = 0
    #print(eval_reward)
    return eval_reward


q_learning()
