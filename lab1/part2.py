import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import os
import sys
import logging

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

# metric viewer
def plot_history(h, *metrics):
    for metric in metrics:
        print(f"{metric}: {h.history[metric][-1]:.4f}")
    figure = plt.figure(figsize=(5 * len(metrics), 3))
    for i, metric in enumerate(metrics, 1):
        ax = figure.add_subplot(1, len(metrics), i)
        ax.xaxis.get_major_locator().set_params(integer=True)
        plt.title(metric)
        plt.plot(h.history[metric], '-')
    plt.show()

# Change the current working directory
os.chdir('./lab1')

# собираем персептрон
model = keras.models.Sequential()
model.add(keras.layers.Dense(
    2, # output space
    activation='sigmoid', # Activation function to use
    input_dim=2, # у персептрона 2 входных нейрона
    kernel_initializer=keras.initializers.RandomNormal(stddev=0.5,mean=0.0),
    bias_initializer=keras.initializers.Zeros()
))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.summary()

#variation 15
points = [[2,-4.7], [-2.3, -4.6], [-4.1,3.2], [1.9,-1.9], [4.5,-4.7],[-0.7, -1.2], [2.6,2.9], [-3.2, -0.2]]
labels = [[1,0],[1,1],[0,1],[1,0],[1,0],[1,0],[0,0],[1,1]]
logging.info(f"points = {points}, len={len(points)}\nlabels = {labels}, len={len(labels)}")

# logging history into file
epochs = 1000
console_out = sys.stdout
log_file = open("history_part2.log", "w")
sys.stdout = log_file # redirect output to file
history = model.fit(x=points, y=labels, batch_size=1, epochs=epochs) # training model
print(history)
plot_history(history, 'mae')
sys.stdout = console_out # back to console out
log_file.close()

# выведем разделяющую прямую
def plot_line(a, b, c):
    xlim, ylim = plt.xlim(), plt.ylim()
    plt.axline((-c / a, 0), slope=-a/b)
    plt.xlim(xlim)
    plt.ylim(ylim)

# поместим точки в массив, распределив по классам
oneClass = []
twoClass = []
treeClass = []
fourClass = []
for point, label in list(zip(points, labels)):
    if label == [0,0]:
        oneClass.append(point)
    if label == [0,1]:
        twoClass.append(point)
    if label == [1,0]:
        treeClass.append(point)
    if label == [1,1]:
        fourClass.append(point)
oneClass = np.array(oneClass)
twoClass = np.array(twoClass)
treeClass = np.array(treeClass)
fourClass = np.array(fourClass)

# размечаем веса
weights = model.get_weights()
w_11, w_12 = weights[0][0][0], weights[0][0][1]
w_21, w_22 = weights[0][1][0], weights[0][1][1]
b1, b2 = weights[1][0], weights[1][1]

plt.scatter(oneClass[:,0], oneClass[:,1], color="b")
plt.scatter(twoClass[:,0], twoClass[:,1], color="orange")
plt.scatter(treeClass[:,0], treeClass[:,1], color="r")
plt.scatter(fourClass[:,0], fourClass[:,1], color="cyan")
plot_line(w_11, w_12, b1)
plot_line(w_21, w_22, b2)
plt.show()