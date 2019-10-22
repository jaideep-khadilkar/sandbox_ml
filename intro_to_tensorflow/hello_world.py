
import keras
import matplotlib.pyplot as plt
import numpy as np

model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer="sgd", loss="mean_squared_error")

xs = np.array([-1.0, 0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

H = model.fit(xs, ys, epochs=500)
epoch_list = range(1, 501)
loss_list = H.history["loss"]

print model.predict([10.0])

plt.scatter(xs, ys, c="r")
plt.plot(xs, model.predict(xs))

# plt.plot(epoch_list, loss_list)
plt.show()
