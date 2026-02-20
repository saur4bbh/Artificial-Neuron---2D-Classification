🧠 Artificial Neuron – 2D Classification

📌 Overview

This project is an interactive Streamlit demo of a single artificial neuron performing 2D binary classification. Users can generate synthetic datasets with multiple Gaussian modes and explore how different activation functions affect classification.
Supported activation functions:

heaviside, logistic, tanh, sin, sign, ReLU, leaky ReLU

The neuron updates its weights using a simple learning rule, and the resulting decision boundary is visualized in real time.


🧠 How It Works

Synthetic data for two classes is generated with configurable modes and samples.

A single neuron is initialized with random weights.

The neuron is trained using the selected activation function.

The decision boundary is plotted over the dataset to show classification regions.


🚀 How to Run

Clone the repository:

git clone (https://github.com/saur4bbh/Artificial-Neuron---2D-Classification)

Navigate into the directory:

cd Artificial-Neuron---2D-Classification

Install dependencies:

pip install -r requirements.txt

Run the Streamlit app:

streamlit run Single_neuron.py


🔧 Features

Generate synthetic 2D datasets with multiple modes per class

Train a single neuron interactively

Select from seven activation functions

Visualize decision boundaries in real time
