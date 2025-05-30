def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output


def train(X, Y, network, loss, loss_prime, epoches=1000, learning_rate=0.05):
    error = 0
    for e in range(epoches):
        for x, y in zip(X, Y):
            # forward
            output = predict(network, x)

            error += loss(y, output)

            # backward
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        error /= len(X)
        if int(e % (epoches/1000)) == 0:
            print("%.1f%% complete, error=%f" % (100 * e/epoches, error))
