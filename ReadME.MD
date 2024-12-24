# Sequence Modeling Project

This project focuses on sequence modeling using Python. It includes various models and techniques for handling sequential data.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Contributing](#contributing)
- [License](#license)

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

To train a model, use the following command:

```bash
python train.py --config config.yaml --model_type LSTM
```

To train the hybrid model, use:

```bash
python atten_train.py --config config.yaml --model_type Hybrid
```

## Models

This project includes the following models:
- LSTM
- Transformer

## Contributing

Contributions are welcome! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.