# Neural Network Image Classification

This repository contains a comprehensive pipeline for building, training, and evaluating neural network models with varying hidden layers to classify images. Implemented using PyTorch, this project showcases the comparison between models with two and three hidden layers, employing L2 regularization and weight initialization to improve performance and reproducibility.

## Project Overview

This project includes the following key components:
- **Data Loading**: Efficiently load image datasets from directories.
- **Preprocessing**: Resize, normalize, and extract features from images.
- **Model Definition**: Define neural network architectures with two and three hidden layers.
- **Training**: Train the models with the specified hyperparameters and save checkpoints.
- **Evaluation**: Evaluate the trained models on a test dataset to measure accuracy.

## Features

- Comparison of neural networks with two and three hidden layers.
- Use of L2 regularization to prevent overfitting.
- Custom weight initialization for better training convergence.
- Detailed training and validation loss and accuracy tracking.

## Getting Started

### Prerequisites

Make sure you have the following installed:
- Python 3.x
- PyTorch
- OpenCV
- NumPy
- Pandas
- Matplotlib

### Installation

1. Clone this repository:
   ```sh
   git clone https://github.com/Albaraazain/neural-network-image-classification.git
   cd neural-network-image-classification
   ```

2. Set up a virtual environment and install dependencies:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

### Usage

1. Prepare your image datasets and organize them into training and testing directories.

2. Run the main script to execute the complete workflow:
   ```sh
   python main.py
   ```

3. View the training and validation loss and accuracy plots to monitor model performance.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
