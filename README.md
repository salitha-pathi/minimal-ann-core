# minimal-ann-core

A minimalistic core implementation of an Artificial Neural Network (ANN) in Python.

## Features

- Lightweight and easy to understand
- No external dependencies
- Suitable for educational purposes and small projects

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/minimal-ann-core.git
cd minimal-ann-core
```

## Usage

```python
from ann import NeuralNetwork

# Define network structure
nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)

# Train the network
nn.train(X_train, y_train, epochs=1000, learning_rate=0.01)

# Make predictions
predictions = nn.predict(X_test)
```

## License

MIT License

## Contributing

Contributions are welcome! Please open issues or submit pull requests.
