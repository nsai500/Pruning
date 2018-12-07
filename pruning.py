import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def weight_prune(weight, k):
	weight_temp = weight
	weight_temp = np.absolute(weight_temp)
	weight_temp = np.sort(weight_temp, axis = None)
	threshold = weight_temp[int(k*weight_temp.size)]
	weight[(weight < threshold) & (weight > -threshold)] = 0
	b = []
	b.append(weight)
	return b

def unit_prune(weight, k):
	weight_temp = np.linalg.norm(weight, axis=0)
	sorted = np.sort(weight_temp)
	threshold = sorted[int(k*sorted.size)]
	j=0
	for i in weight_temp:
		if(i<threshold):
			weight[:, j] = 0
		j=j+1
	b = []
	b.append(weight)
	return b

input = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = input.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(1000, activation=tf.nn.relu, use_bias=False),
  tf.keras.layers.Dense(1000, activation=tf.nn.relu, use_bias=False),
  tf.keras.layers.Dense(500, activation=tf.nn.relu, use_bias=False),
  tf.keras.layers.Dense(200, activation=tf.nn.relu, use_bias=False),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax, use_bias=False)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

#model.fit(x_train, y_train, epochs=10)
#model.save_weights('pre_trained.h5', save_format='h5')

model.load_weights('pre_trained.h5')

results = []
x=[]
y=[]
per = [0, 25, 50, 60, 70, 80, 90, 95, 97, 99]

for k2 in per:
	k = k2*0.01
	for i in range(2,5):
		weight = model.layers[i].get_weights()[0]
		model.layers[i].set_weights(weight_prune(weight, k))
	score, acc = model.evaluate(x_test, y_test)
	results.append("k = " + str(k) + " accuracy = " + str(acc))
	x.append(k*100)
	y.append(acc*100)
	model.load_weights('pre_trained.h5')

for result in results:
	print(result)

plt.plot(x, y)
plt.xlabel('Sparsity %')
plt.ylabel('Accuracy %')
plt.title('Weight Pruning')
plt.show()

model.load_weights('pre_trained.h5')

results = []
x=[]
y=[]
per = [0, 25, 50, 60, 70, 80, 90, 95, 97, 99]

for k2 in per:
	k = k2*0.01
	for i in range(2,5):
		weight = model.layers[i].get_weights()[0]
		model.layers[i].set_weights(unit_prune(weight, k))
	score, acc = model.evaluate(x_test, y_test)
	results.append("k = " + str(k) + " accuracy = " + str(acc))
	x.append(k*100)
	y.append(acc*100)
	model.load_weights('pre_trained.h5')

for result in results:
	print(result)

plt.plot(x, y)
plt.xlabel('Sparsity %')
plt.ylabel('Accuracy %')
plt.title('Unit Pruning')
plt.show()