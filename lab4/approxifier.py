import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from numpy import sin, cos, pi
from sklearn.model_selection import train_test_split
from keras import backend
from matplotlib.colors import LinearSegmentedColormap

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px


# RBFlayer from lection
class RBFLayer(keras.layers.Layer):
    def __init__(self, output_dim, mu_init='uniform', sigma_init='random_normal', **kwargs):
        self.output_dim = output_dim
        self.mu_init = mu_init
        self.sigma_init = sigma_init
        super(RBFLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.mu = self.add_weight(name='mu', shape=(input_shape[1], self.output_dim), initializer=self.mu_init, trainable=True)
        self.sigma = self.add_weight(name='sigma', shape=(self.output_dim,), initializer=self.sigma_init, trainable=True)
        super(RBFLayer, self).build(input_shape)
        
    def call(self, inputs):
        diff = backend.expand_dims(inputs) - self.mu
        output = backend.exp(backend.sum(diff ** 2, axis=1) * self.sigma)
        return output


def plot_history(h, *metrics):
    "function for showing model history on plot with metrics"
    for metric in metrics:
        print(f"{metric}: {h.history[metric][-1]:.4f}")
    figure = make_subplots(rows=1, cols=len(metrics), subplot_titles=metrics)
    for i, metric in enumerate(metrics, 1):
        ax = np.arange(0, len(h.history[metric]), 1)
        figure.add_trace(
            go.Scatter(x=ax, y=h.history[metric], name=metric), col=i, row=1
        )
    figure.update_layout(title_text="Метрики")
    figure.show()

def plot_three_classes(data, labels, colors):
    plt.scatter(data[:, 0], data[:, 1], c=[colors[i[1]+i[2]*2] for i in labels], marker='.')
COLORS = ['red', 'green', 'blue']


def f(t):
    return cos(-2 * t**2 + 7 *t)

h = 0.01
train_data = np.arange(0, 3.5, h)
train_labels = f(train_data)

df = pd.DataFrame({
    "train_data": train_data,
    "train_labels": train_labels,
    "type": "train"
})

px.scatter(df, x="train_data", y="train_labels", title="Функция").show()

# generating model
model = keras.models.Sequential([
    RBFLayer(16, input_dim=1, mu_init=keras.initializers.RandomUniform(minval = 0, maxval = 5)),
    keras.layers.Dense(7, activation='tanh'),
    keras.layers.Dense(16, activation='tanh'),
    keras.layers.Dense(4, activation='tanh'),
    keras.layers.Dense(1, activation='linear')
])
model.compile(keras.optimizers.Adam(3e-4), 'mse', ['mae'])

hist = model.fit(train_data, train_labels, batch_size=7, epochs=200, verbose=0, shuffle=True)

plot_history(hist, 'loss', 'mae')

trained = pd.DataFrame({
    "train_data": train_data,
    "train_labels": model.predict(train_data).flat,
    "type": "prediction"
})

mu = model.get_layer(index=0).get_weights()[0][0]
mu_y = model.predict(mu)

fig = px.scatter(df, x="train_data", y="train_labels", title="Функция")
fig.add_trace(go.Scatter(x=trained["train_data"], y=trained["train_labels"], name="predicted"))
fig.add_trace(go.Scatter(x=mu, y=mu_y.flat, name="mu", mode="markers"))

fig.show()