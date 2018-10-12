from sklearn import metrics, preprocessing
import numpy as np


# we assume data is distributed normally
def processing_data(data):
    # Normalization
    # https://datascience.stackexchange.com/questions/12321/difference-between-fit-and-fit-transform-in-scikit-learn-models
    scaler = preprocessing.StandardScaler()
    data = data.astype(str).astype(int)
    price_data = data.get('Price')
    time_data = data.get('Time')
    price_diff = np.diff(price_data.values)
    price_diff = np.insert(price_diff, 0, 0)

    # Preprocess data, normalization
    # column normalize the following: [time price_difference]
    xdata = np.column_stack((time_data, price_diff))
    xdata = np.nan_to_num(xdata)

    xdata = scaler.fit_transform(xdata)
    state = xdata[0:1, :]

    return state, xdata
