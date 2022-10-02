import numpy as np
from tensorflow import keras
import plotly.express as px
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os
import sys
import logging
import pandas as pd

# initializing logging
yellow = "\x1b[33;20m"
reset = "\x1b[0m"
logging.basicConfig(
    level=logging.INFO,
    format=yellow + "[%(levelname)s] %(message)s" + reset,
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def plot_history(h, *metrics):
    "function for showing model history on plot with metrics"
    for metric in metrics:
        print(f"{metric}: {h.history[metric][-1]:.4f}")
    figure = make_subplots(rows=1, cols=len(metrics), subplot_titles=metrics)
    for i, metric in enumerate(metrics,1):
        ax = np.arange(0, len(h.history[metric]),1)
        figure.add_trace(
            go.Scatter(x=ax, y=h.history[metric]), col=i, row=1
        )
    figure.update_layout(title_text="Метрики")
    figure.show()

# Change the current working directory
os.chdir('./lab2')

# configurations
window = 5
t = np.arange(1,3,0.01)
signal = lambda t: np.sin(np.sin(t)*t**3 - 10)

data = signal(t)
target = data[window:]
data = np.array([data[i:i+window] for i in range(0, len(data) - window)])

# show signal
signal_show_df = pd.DataFrame({
    "signal": signal(t),
    "t": t
})
fig = px.line(signal_show_df, x="t", y="signal", title="Показ сигнала")
fig.show()

# configurating model
model = keras.models.Sequential([
    keras.layers.Dense(1, input_dim=window, activation='linear')
])
model.compile(keras.optimizers.SGD(0.01), 'mse', ['mse'])
hist = model.fit(data, target, batch_size=1, epochs=50, verbose=0, shuffle=True)

# показываем метрики
plot_history(hist, "mse")


pred = model.predict(data)
logging.info(f"target {target[:5]}, pred {pred[:5]}")

finalFig = make_subplots(cols=2, rows=1, subplot_titles=["prediction", "errors"])

finalFig.add_trace(go.Scatter(x=t[window:], y=target,name="signal"), col=1, row=1)
finalFig.add_trace(go.Scatter(x=t[window:], y=pred[:,0],name="predicted"), col=1, row=1)

finalFig.add_trace(go.Scatter(x=t[window:], y=target - pred[:,0], name="error"), col=2, row=1)
finalFig.add_trace(go.Scatter(x=t[window:], y=[0]*len(t[window:]), name="ideal_error"), col=2, row=1)

finalFig.show()