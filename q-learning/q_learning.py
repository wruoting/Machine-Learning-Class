import tensorflow as tf
from csv_to_dataframe import csv_to_dataframe
from graph import graph, map_to_graph
import numpy as np
import pandas as pd
from preprocessing import processing_data
from keras.models import load_model
from q_model import q_model
import random


def q_learning(model=None):
    df_data = csv_to_dataframe('sin-test.csv')
    batch_range = range(0, 15)
    state, price_data = processing_data(df_data, batch_range)
        # Output:[[ Price Time ]]
        # State: [[ 1 2 ]]
        # Data: [[ 1 2 ]]
        #       [[ 3 4 ]]
        #       ...

    decision_state = pd.Series(index=np.arange(len(price_data))) # the length of the data
        # Output: 0     NaN
        # 1     NaN
        # 2     NaN
        # 3     NaN

    stored_buffer = 50
    epochs = 300
    gamma = 0.9  # discount factor, we are rewarding long term trading with this high factor
    status = 1
    terminal_state = False
    learning_progress = []
    batch_size = len(batch_range)
    replay = []
    state_index = 0
    loss_matrix = []
    for epoch in range(epochs):
        print('next')
        while status == 1:
            # We start in state S
            # Run the Q function on S and get the predicted value on Q
            q_value = model.predict(state[state_index], batch_size=batch_size)
            q_value = q_value[0][0]

            # Softmax it
            q_value_list, softmax_percentages = softmax(q_value, state_index, len(state))
            action = int(np.random.choice(3, 1, p=softmax_percentages))

            # take action and return new state
            state_index, decision_state, terminal_state = take_action(action, decision_state, state, state_index, stored_buffer, batch_range)
            # observe reward
            reward = get_reward(state, state_index, action, price_data, decision_state, batch_size)
            # Store states if less than our buffer
            # Train on a random subset of training sets in our buffer
            if len(replay) <= stored_buffer and not terminal_state:
                replay.append((state, action, reward, state_index))
            else:
                batches = len(state)-1
                mini_batch = []
                if batches <= len(replay):
                    mini_batch = random.sample(replay, batches)
                elif batches > len(replay):
                    mini_batch = random.sample(replay, len(replay))
                x_train = []
                y_train = []
                for batch in mini_batch:
                    # Get max_Q(S',a)
                    state, action, reward, new_state_index = batch
                    if new_state_index < batches:
                        old_q_val = model.predict(state[new_state_index], batch_size=batch_size)
                        new_q_val = model.predict(state[new_state_index+1], batch_size=batch_size)
                        max_q = np.max(new_q_val)
                        print('hi')
                        print(action)
                        print(old_q_val)
                        y = old_q_val[0][0]
                        update = reward + gamma * max_q - y[action]
                        y[action] += update
                        print(y)
                    elif new_state_index == batches:
                        q_val = model.predict(state[new_state_index], batch_size=batch_size)
                        max_q = np.max(q_val)
                        y = q_val[0][0]
                        y[action] = max_q
                    if new_state_index <= batches:
                        x_train.append(state[new_state_index])
                        y_train.append(y.reshape(1, 1, 3))

                for (x, y) in zip(x_train, y_train):
                    model.fit(x, y, batch_size=1, epochs=1, verbose=0)
                    loss, mse = model.evaluate(x, y, verbose=0)
                    loss_matrix.append(loss)

            if terminal_state:  # if reached terminal state, update epoch status
                status = 0
        # Pick a random index to start from
        state_index = np.random.randint(0,len(state)-1, size=1)[0]
        status = 1
        replay = []
    # save your model
    model.save('q_model.h5')
    return loss_matrix


# we're not going to use an anneal value because this is just a proportion of q values
# an annealing value should be used closer to the end of a sequence, we can inject state here
def softmax(q_values, state_index, total_steps):
    q_value_softmax = []
    q_value_list = []
    if state_index == 0:
        annealing_factor = 0.01 # very small factor for now
    else:
        annealing_factor = state_index/total_steps
    for value in q_values:
        if value < 0:
            q_values += np.multiply(-1, value)
    # we deal with overflow on exponential annealing factor issues by just giving the biggest q value the choice
    try:
        np.seterr(over='raise')
        sum_exponent = np.sum(np.exp(np.divide(q_values, annealing_factor)))
        for value in q_values:
            numerator = np.exp(np.divide(value, annealing_factor))
            q_value_softmax.append(np.divide(numerator,sum_exponent))
            q_value_list.append(value)
    except Exception as e:
        maximum = 0
        final_index = 0
        for index, value in enumerate(q_values):
            if value > maximum:
                maximum = value
                final_index = index

        q_value_softmax = [1 if index == final_index else 0 for index, value in enumerate(q_values)]
        pass

    # because of mtrand's high tolerance, we're gonna do something really janky here
    sum_tolerance = np.sum(np.round(np.multiply(q_value_softmax,np.power(10,9)),0))
    new_tolerance = np.round(np.multiply(q_value_softmax,np.power(10,9)),0)
    tolerance_factor = np.subtract(sum_tolerance, np.power(10,9))
    # just add it to the first element, we don't really care for a 10e-8 impact
    tolerance_measured = False
    for index, val in enumerate(new_tolerance):
        if val > 0 and np.subtract(val,tolerance_factor) >= 0 and not tolerance_measured:
            new_tolerance[index] = np.subtract(val,tolerance_factor)
            tolerance_measured = True
    adjusted_q_value_softmax = np.divide([new_tolerance[0],new_tolerance[1],new_tolerance[2]], np.power(10,9))
    return q_value_list, adjusted_q_value_softmax


def take_action(action, decision_state, state, state_index, stored_buffer, batch_range, eval=False):
    terminal_state = False
    # if the current state is the last point in the frame
    if eval:
        max_length = len(state)
    else:
        max_length = len(state) if len(state) <= stored_buffer else stored_buffer
    time_step_length = len(batch_range)
    state_index += 1
    if state_index >= max_length:
        terminal_state = True
        return state_index, decision_state, terminal_state
    decision_state_index = state_index + time_step_length
    #take action
    if action == 1: #buy
        decision_state.loc[decision_state_index] = 1
    elif action == 2: #sell
        decision_state.loc[decision_state_index] = -1
    else: #hold
        decision_state.loc[decision_state_index] = 0

    return state_index, decision_state, terminal_state


def get_reward(state, state_index, action, data, decision_state, batch_size):
    decision_state.fillna(value=0, inplace=True)
    time_step_length = batch_size
    current_index = state_index + time_step_length
    past_index = state_index + time_step_length-1
    total_gains = 0
    for index in range(past_index, current_index):
        total_gains += data[index][1]
    buy_reward = total_gains
    hold_reward = 0
    sell_reward = np.multiply(total_gains, -1)
    action_to_reward = {
        0: hold_reward,
        1: buy_reward,
        2: sell_reward
    }
    return action_to_reward[action]


def evaluate_q_epoch(state, price_data, model, decision_state, batch_range, stored_buffer):
    eval_reward = 0
    terminal_state = False
    state_index = 0
    max_length = len(state)
    batch_size = len(batch_range)
    while not terminal_state:
        reshaped_data = np.reshape(state[state_index],(1,1,np.multiply(len(batch_range),2)))
        q_value = model.predict(reshaped_data, batch_size=1)
        q_value = q_value[0][0]
        action = np.argmax(q_value)
        print(action)
        state_index, decision_state, terminal_state = take_action(action, decision_state, state, state_index, stored_buffer, batch_range, eval=True)
        eval_reward += get_reward(state, state_index, action, price_data, decision_state, batch_size)
        if state_index < max_length - 1:
            state_index += 1
        else:
            terminal_state = True
    return eval_reward, decision_state


q_learning(model=q_model())
