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

# Change the current working directory
os.chdir('.\\lab1')

# собираем персептрон
model = keras.models.Sequential()
model.add(keras.layers.Dense(
    1, # output space
    activation='sigmoid', # Activation function to use
    input_dim=2, # у персептрона 2 входных нейрона
    kernel_initializer=keras.initializers.RandomNormal(stddev=0.5,mean=0.0),
    bias_initializer=keras.initializers.Zeros()
))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.summary()


print("\ninitiating entering points")
# variation 4
points = np.array([[-4,-3.6], [-3.4, 1.2], [0.7, -4.5], [4.3, 2.2], [2.3, -4.4], [3.6, 4.3]])
labels = np.array([0, 1, 0, 0, 0, 1])
print(f"points = {points}\nlabels = {labels}")

# logging history into file
console_out = sys.stdout
log_file = open("history.log", "w")
sys.stdout = log_file # redirect output to file
model.fit(x=points, y=labels, batch_size=1, epochs=100) # training model
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
def liner_func(x, w1, w2, b):
    return -x*w1/w2 - b*w2

def prepare(xmin, xmax, w1,w2,b):
    X = [xmin, xmax]
    Y = [liner_func(xmin,w1,w2,b), liner_func(xmax,w1,w2,b)]
    return X,Y
line_x, line_y = prepare(np.amin(points[:, 0]), np.amax(points[:, 0]),w1,w2,b)

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
plt.plot(line_x, line_y, color="orange")
plt.show()
