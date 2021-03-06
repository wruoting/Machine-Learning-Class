from sklearn import metrics, preprocessing
import numpy as np
import pandas as pd


# we assume data is distributed normally
def processing_data(data, batch_range):
    scaler = preprocessing.StandardScaler()
    data = data.astype(str).astype(float)
    price_data = data.get('Price')
    time_data = data.get('Time')
    price_diff = np.diff(price_data.values)
    price_diff = np.insert(price_diff, 0, 0)
    batch_size = len(batch_range)

    # Preprocess data, normalization
    # column normalize the following: [time price_difference]
    xdata = np.column_stack((time_data, price_diff))
    xdata = np.nan_to_num(xdata)
    unscaled_data = xdata
    scaled_data = scaler.fit_transform(xdata)
    state = []
    # slice into states
    num_of_slices = int(np.divide((len(xdata)-len(xdata)%batch_size), batch_size))

    for index in range(0, len(xdata)-batch_size):
        next_index = index + batch_size
        # Squeezing a range of data pairs into one dimension to output as one batch
        reshaped_data = np.reshape(scaled_data[index:next_index, 0:2], (1, 1, np.multiply(len(batch_range), 2)))
        state.append(reshaped_data)
    return state, scaled_data, unscaled_data
