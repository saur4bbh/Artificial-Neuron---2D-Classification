import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


class ArtificialNeuron:
    def __init__(self, lr=0.1, activation="heaviside", beta=1.0):
        self.w = np.random.randn(3) * 0.1 
        self.lr = lr
        self.activation = activation
        self.beta = beta

    def net(self, x):
        return np.dot(self.w, np.append(x, 1.0)) 

    def activate(self, s):
        if self.activation == "heaviside":
            return 1 if s >= 0 else 0

        elif self.activation == "logistic":
            return 1 / (1 + np.exp(-self.beta * s))

        elif self.activation == "sin":
            return np.sin(s)

        elif self.activation == "tanh":
            return np.tanh(s)

        elif self.activation == "sign":
            return 1 if s > 0 else -1 if s < 0 else 0

        elif self.activation == "relu":
            return max(0, s)

        elif self.activation == "lrelu":
            return s if s > 0 else 0.01 * s

    def derivative(self, s):
        if self.activation == "heaviside":
            return 1 

        elif self.activation == "logistic":
            y = self.activate(s)
            return self.beta * y * (1 - y)

        elif self.activation == "sin":
            return np.cos(s)

        elif self.activation == "tanh":
            return 1 - np.tanh(s) ** 2 

        else:
            return 0

    def predict(self, x):
        return self.activate(self.net(x)) 

    def train(self, X, d, epochs=50):
        for _ in range(epochs):
            for x, target in zip(X, d):
                s = self.net(x)
                y = self.activate(s)
                self.w += self.lr * (target - y) * self.derivative(s) * np.append(x, 1.0)



def generate_class(modes, samples, label):
    X, y = [], []
    for _ in range(modes):
        mean = np.random.uniform(-1, 1, size=2)
        cov = np.diag(np.random.uniform(0.05, 0.2, size=2))
        pts = np.random.multivariate_normal(mean, cov, samples)
        X.append(pts)
        y.extend([label] * samples)
    return np.vstack(X), np.array(y)

def generate_data(modes0, modes1, samples): 
    X0, y0 = generate_class(modes0, samples, 0)
    X1, y1 = generate_class(modes1, samples, 1)
    return np.vstack([X0, X1]), np.concatenate([y0, y1])


def plot_decision_boundary(neuron, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid( 
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    ) 

    grid = np.c_[xx.ravel(), yy.ravel()] 
    Z = np.array([neuron.predict(p) for p in grid]) 

    ''' #For clear separation with two bg colours
    if neuron.activation in ["tanh", "sin"]: 
        Z = (Z > 0).astype(int)
    else:
        Z = (Z > 0.5).astype(int)
    '''
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c="blue", label="Class 0")
    ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c="red", label="Class 1")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.set_title("Neuron Decision Boundary")

    return fig


st.set_page_config(page_title="Artificial Neuron", layout="centered")
st.title("Artificial Neuron – 2D Classification")

activation = st.selectbox(
    "Activation function",
    ["heaviside", "logistic", "sin", "tanh", "sign", "relu", "lrelu"]
)

learning_rate = st.slider("Learning rate η", 0.001, 1.0, 0.01)
modes_class0 = st.number_input("Number of modes – Class 0", 1, 5, 1)
modes_class1 = st.number_input("Number of modes – Class 1", 1, 5, 1)
samples_per_mode = st.number_input("Samples per mode", 10, 500, 100)

if st.button("Generate data & train neuron"):
    X, y = generate_data(modes_class0, modes_class1, samples_per_mode)

    neuron = ArtificialNeuron(
        lr=learning_rate,
        activation=activation
    )

    if activation in ["heaviside", "logistic", "sin", "tanh"]:
        neuron.train(X, y)

    fig = plot_decision_boundary(neuron, X, y)
    st.pyplot(fig)