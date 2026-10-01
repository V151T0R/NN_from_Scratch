def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output