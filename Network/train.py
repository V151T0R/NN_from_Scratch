from Network.predict import predict


def train(network, loss, loss_prime, x_train, y_train, epochs=1000, learning_rate=0.01, verbose=True):
    sample_count = len(x_train)
    if sample_count == 0:
        raise ValueError("Training data cannot be empty")
    if epochs < 1:
        raise ValueError("epochs must be greater than zero")

    if verbose:
        print(
            f"Training started: samples={sample_count}, "
            f"epochs={epochs}, learning_rate={learning_rate}"
        )

    for epoch in range(epochs):
        total_loss = 0
        for x, y in zip(x_train, y_train):     #x is the input, y is the target output
            # forward
            output = predict(network, x)

            # error
            total_loss += loss(y, output)

            # backward
            gradient = loss_prime(y, output)
            for layer in reversed(network):
                gradient = layer.backward(gradient, learning_rate)

        average_loss = total_loss / sample_count
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs} - loss: {average_loss:.6f}")

    if verbose:
        print(f"Training complete: final_loss={average_loss:.6f}")