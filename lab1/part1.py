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
    1, # output space
    activation='sigmoid', # Activation function to use
    input_dim=2, # у персептрона 2 входных нейрона
    kernel_initializer=keras.initializers.RandomNormal(stddev=0.5,mean=0.0),
    bias_initializer=keras.initializers.Zeros()
))

model.compile(loss='mse', optimizer='adam', metrics=['mae', 'accuracy'])
model.summary()


print("\ninitiating entering points")
# variation 15
points = np.array([[-4.1,-2.4], [-1.7, 1.7], [-3.7, 2.2], [-4, 1.5], [-0.1, 2.7], [2.1, 4]])
labels = np.array([1, 1, 0, 0, 1, 1])
print(f"points = {points}\nlabels = {labels}")

# logging history into file
epochs = 1000
console_out = sys.stdout
log_file = open("history.log", "w")
sys.stdout = log_file # redirect output to file
history = model.fit(x=points, y=labels, batch_size=1, epochs=epochs) # training model
print(history)
plot_history(history, 'accuracy', 'mae')
sys.stdout = console_out # back to console out
log_file.close()

# запишем веса в переменные
print(f"\ninitializing weights...")
weights = model.layers[0].get_weights()
w1, w2 = weights[0][0][0], weights[0][1][0]
b = weights[1][0] 
logging.info(f"weights = {weights}")
logging.info(f"w1 = {w1}, w2 = {w2}")
logging.info(f"b = {b}")

# выведем разделяющую прямую
def plot_line(a, b, c):
    xlim, ylim = plt.xlim(), plt.ylim()
    plt.axline((-c / a, 0), slope=-a/b)
    plt.xlim(xlim)
    plt.ylim(ylim)

# поместим точки в массив, распределив по классам
oneClass = []
twoClass = []
for point, label in list(zip(points, labels)):
    if label == 1:
        twoClass.append(point)
    if label == 0:
        oneClass.append(point)
oneClass = np.array(oneClass)
twoClass = np.array(twoClass)

# рисуем график
plt.scatter(oneClass[:,0], oneClass[:,1], color="b")
plt.scatter(twoClass[:,0], twoClass[:,1], color="r")
plot_line(w1,w2, b)
plt.show()
