def sgd(X, y, learning_rate, num_epochs, batch_size):
    # initialize the model parameters
    w = np.zeros(X.shape[1])
    b = 0
    
    for epoch in range(num_epochs):
        # shuffle the training data
        X, y = shuffle(X, y)
        
        # divide the training data into mini-batches
        for i in range(0, X.shape[0], batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]
            
            # compute the predicted output
            y_pred = np.dot(X_batch, w) + b
            
            # update the model parameters
            w -= learning_rate * (y_pred - y_batch) * X_batch
            b -= learning_rate * (y_pred - y_batch)
    
    return w, b
