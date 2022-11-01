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


def ellipse(t, a, b, x0, y0, alpha):
    x = a * cos(t)
    y = b * sin(t)
    x, y = rotate(x, y, alpha)
    return np.array((x + x0, y + y0)).T


def parabola(t, p, x0, y0, alpha):
    x = x0 + t ** 2 / (2. * p)
    y = y0 + t
    x, y = rotate(x, y, alpha)
    return np.array((x, y)).T


def rotate(x, y, alpha):
    xr = x * cos(alpha) - y * sin(alpha)
    yr = x * sin(alpha) + y * cos(alpha)
    return xr, yr


# autopep8: off
a1 = 0.4; b1 = 0.15; alpha1 = pi/6;   x01 = 0;    y01 = 0
a2 = 0.7; b2 = 0.5;  alpha2 = pi/3;   x02 = 0;    y02 = 0
p3 = 1;               alpha3 = 0;      x03 = 0;    y03 = -0.8
# autopep8: on

t = np.arange(0, 2 * pi, 0.025)
ellipse1 = ellipse(t, a1, b1, x01, y01, alpha1)
ellipse2 = ellipse(t, a2, b2, x02, y02, alpha2)
parabola3 = parabola(t[:len(t)//4], p3, x03, y03, alpha3) #! сделать пометку в отчете почему тут //4

figures = pd.DataFrame({
    "y": ellipse1[:, 0],
    "t": ellipse1[:, 1],
    "type": "ellipse1"
})
figures = pd.concat([figures, pd.DataFrame({
    "y": ellipse2[:, 0],
    "t": ellipse2[:, 1],
    "type": "ellipse2"
})])

figures = pd.concat([figures, pd.DataFrame({
    "y": parabola3[:, 0],
    "t": parabola3[:, 1],
    "type": "parabola"
})])

fig = px.line(figures, x="t", y="y", title="Фигуры", color="type")
fig.show()

rng = np.random.default_rng()

data = np.array((*rng.choice(ellipse1, 120, False, axis=0),
                  *rng.choice(ellipse2, 100, False, axis=0),
                  *rng.choice(parabola3, 60, False, axis=0)))
labels = np.array((*[[1, 0, 0] for _ in range(120)],
                    *[[0, 1, 0] for _ in range(100)],
                    *[[0, 0, 1] for _ in range(60)]))


df = pd.DataFrame({
    "y": data[:120,0],
    "t": data[:120,1],
    "type": "first_class",
})
df = pd.concat([df, pd.DataFrame({
    "y": data[120:220,0],
    "t": data[120:220,1],
    "type": "second_class",
}), pd.DataFrame({
    "y": data[220:,0],
    "t": data[220:,1],
    "type": "third_class",
})])

fig = px.scatter(df, x="t", y="y", title="Точки обучения", color="type")
fig.show()

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, train_size=0.8)

model = keras.models.Sequential([
    keras.layers.Dense(20, input_dim=2, activation='tanh'),
    keras.layers.Dense(50, activation='tanh'),
    keras.layers.Dense(3, activation='sigmoid')
])
model.compile(keras.optimizers.Adam(0.01), 'mse', ['accuracy'])

hist = model.fit(train_data, train_labels, validation_data=(test_data, test_labels), batch_size=20, epochs=100, verbose=0)

plot_history(hist, 'loss', 'accuracy', 'val_accuracy')

n = 100
x = np.linspace(-1.2, 1.2, n)
y = np.linspace(-1.2, 1.2, n)

xv, yv = np.meshgrid(x, y)
z = model.predict(np.c_[xv.ravel(), yv.ravel()]).argmax(axis=1).reshape(n, n)

px.imshow(z).show()