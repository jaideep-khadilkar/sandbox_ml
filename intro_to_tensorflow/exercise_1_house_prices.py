
import keras
import matplotlib.pyplot as plt
import numpy as np

model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer="sgd", loss="mean_squared_error")

xs = np.array([0, 1, 2, 3, 4, 5, 6], dtype=float)
ys = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)

H = model.fit(xs, ys, epochs=500)
epoch_list = range(1, 501)
loss_list = H.history["loss"]

print model.predict([10.0])

plt.scatter(xs, ys, c="r")
plt.plot(xs, model.predict(xs))

# plt.plot(epoch_list, loss_list)
plt.show()
