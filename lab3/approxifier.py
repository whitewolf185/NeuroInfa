import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from numpy import sin, cos, pi
from sklearn.model_selection import train_test_split
from matplotlib.colors import LinearSegmentedColormap

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px


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
    keras.layers.Dense(50, input_dim=1, activation='tanh'),
    keras.layers.Dense(100, activation='tanh'),
    keras.layers.Dense(15, activation='tanh'),
    keras.layers.Dense(1, activation='linear')
])
model.compile(keras.optimizers.Adam(0.001), 'mse', ['mae'])

hist = model.fit(train_data, train_labels, batch_size=7, epochs=500, verbose=0, shuffle=True)

plot_history(hist, 'loss', 'mae')

trained = pd.DataFrame({
    "train_data": train_data,
    "train_labels": model.predict(train_data).flat,
    "type": "prediction"
})


fig = px.scatter(df, x="train_data", y="train_labels", title="Функция")
fig.add_trace(go.Scatter(x=trained["train_data"], y=trained["train_labels"], name="predicted"))

fig.show()