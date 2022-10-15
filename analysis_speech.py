import numpy as np
import pvml
import matplotlib.pyplot as plt
import random
import Feature_normalization as norm


words = open("classes.txt").read().split()

Xtest, Ytest = np.load("test.npz").values()
print("Test data", Xtest.shape, Ytest.shape)

Xtrain, Ytrain = np.load("train.npz").values()
print("Train data", Xtrain.shape, Ytrain.shape)

# normalize the data
Xtrain, Xtest = norm.meanvar_normalization(Xtrain, Xtest)


# load the trained neural network
net = pvml.MLP.load("speech_mlp.npz")

labels, probs = net.inference(Xtest)
w = net.weights[0]


# Drawing the set of weights of a given class
n = 34 
image = w[:, n].reshape(20, 80)
maxval = np.abs(image).max()
plt.imshow(image, cmap="seismic", vmin= -maxval, vmax= maxval)
plt.title(words[n])
plt.colorbar()
plt.show()

fig, axes = plt.subplots(2, 2, figsize = (10, 3))
for ax in axes.flat:
 	i = random.randint(0, 34)
 	pixels = w[:, i].reshape(20, 80)
 	maxval = np.abs(pixels).max()
 	image = ax.imshow(pixels, cmap="seismic", vmin= -maxval, vmax= maxval)
 	ax.set(xlabel='Time periods', ylabel='Frequencies', title= words[Ytrain[i]])
 	ax.label_outer()
fig.colorbar(image, ax=axes.ravel().tolist())
plt.show()


# Reduce the data in case it is needed
n = 35
Xtrain 	= Xtrain[Ytrain < n, :]
Ytrain 	= Ytrain[Ytrain < n]
Xtest 	= Xtest[Ytest < n, :]
Ytest 	= Ytest[Ytest < n]

# confusion matrix
cm = np.zeros(( n, n )) 

for i in range(Xtest.shape[0]):
	cm[Ytest[i], labels[i]] += 1

# this normalizes each row by its total sum
cm /= cm.sum(1, keepdims=True)

print("\t \t \t", end="")
for j in range(n):
	print(words[j][:5], end="  \t")
print()
for i in range(n):
	print("%9s" % words[i], end="\t")
	for j in range(n):
		print("%3.0f " % (cm[i, j] * 100), end="\t")
	print()

plt.imshow(cm, cmap= "inferno")
# for i in range(n):
# 	for j in range (n):
# 		plt.text(j, i, int(100 * cm[i, j]), color = "pink")
plt.xticks(range(n), words[-n:], rotation = 60)
plt.xlabel("Labels assigned by the net")
plt.yticks(range(n), words[-n:])
plt.ylabel("Real Classes")
plt.title("Confusion Matrix")
plt.colorbar()
plt.show()
