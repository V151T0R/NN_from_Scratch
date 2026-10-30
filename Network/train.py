# Import the loss layer class used to compute loss values and their gradients.
from Layer.LossLayer import LossLayer
# Import the prediction function to compute network outputs for given inputs.
from Network.predict import predict


# Define the training function that optimizes the network over the dataset.
def train(network, loss, loss_prime, x_train, y_train, epochs=1000, learning_rate=0.01, verbose=True):
    # Count how many training samples are available.
    sample_count = len(x_train)
    # Ensure there is at least one training sample before training begins.
    if sample_count == 0:
        raise ValueError("Training data cannot be empty")
    # Ensure the number of epochs is a valid positive integer.
    if epochs < 1:
        raise ValueError("epochs must be greater than zero")

    # Create the loss layer that contains the loss function and its derivative.
    loss_layer = LossLayer(loss, loss_prime)

    # If verbose mode is enabled, print a summary before training starts.
    if verbose:
        print(
            f"Training started: samples={sample_count}, "
            f"epochs={epochs}, learning_rate={learning_rate}"
        )

    # Train the network for the specified number of epochs.
    for epoch in range(epochs):
        # Reset the total loss for this epoch.
        total_loss = 0
        # Iterate over each input-label pair in the training set.
        for x, y in zip(x_train, y_train):
            # Run a forward pass to get the model prediction for this sample.
            output = predict(network, x)
            # Add the loss for this sample to the epoch's total.
            total_loss += loss_layer.forward(output, y)
            # Compute the gradient of the loss with respect to the last layer output.
            gradient = loss_layer.backward()  # manual call to backward() to get the gradient of the loss with respect to the last layer's output
            # Propagate the gradient backward through the network and update weights.
            for layer in reversed(network):
                gradient = layer.backward(gradient, learning_rate)

        # Calculate the average loss across all samples for this epoch.
        average_loss = total_loss / sample_count
        # If verbose mode is enabled, print the current epoch loss.
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs} - loss: {average_loss:.6f}")

    # If verbose mode is enabled, print the final training loss after all epochs.
    if verbose:
        print(f"Training complete: final_loss={average_loss:.6f}")