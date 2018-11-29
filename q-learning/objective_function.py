from q_learning import q_learning
def objective(params):
    """Objective function to minimize with smarter return values"""


    q_learning(model=model_set_params)
    # Create the model object
    f = np.poly1d([1, -2, -28, 28, 12, -26, 100])

    # Evaluate the function
    start = timer()
    loss = f(x) * 0.05
    end = timer()

    # Calculate time to evaluate
    time_elapsed = end - start

    results = {'loss': loss, 'status': STATUS_OK, 'x': x, 'time': time_elapsed}

    # Return dictionary
    return results