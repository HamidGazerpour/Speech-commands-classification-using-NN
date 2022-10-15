import numpy as np
import pvml
import matplotlib.pyplot as plt
import random
import Feature_normalization as norm

words = open("classes.txt").read().split()

Xtrain, Ytrain = np.load("train.npz").values()
# print("Train data", Xtrain.shape, Ytrain.shape)
# print(np.bincount(Ytrain))

Xtest, Ytest = np.load("test.npz").values()
# print("Test data", Xtest.shape, Ytest.shape)
# print(np.bincount(Ytest))

# # draw the spectrogram of one sample
# id = 762
# image = Xtrain[id, :].reshape(20, 80)
# fig, axes = plt.subplots(2, figsize= (6,2))
# for ax in axes.flat:
# 	i = random.randint(0, 84290)
# 	im = ax.imshow(Xtrain[i, :].reshape(20, 80), cmap= "inferno")
# 	ax.set(xlabel='Time periods', ylabel='Frequencies', title= words[Ytrain[i]])
# 	ax.label_outer()
# fig.colorbar(im, ax=axes.ravel().tolist())
# plt.show()



# here we perform some feature normalization
#Xtrain, Xtest = norm.minmax_normalization(Xtrain, Xtest)
Xtrain = norm.l2_normalization(Xtrain)
Xtest = norm.l2_normalization(Xtest)

# Reduce the data to class 0-n
n = 35
Xtrain 	= Xtrain[Ytrain < n, :]
Ytrain 	= Ytrain[Ytrain < n]
Xtest 	= Xtest[Ytest < n, :]
Ytest 	= Ytest[Ytest < n]


def accuracy(net, X, Y):
	labels, probs = net.inference(X)
	acc = (labels == Y).mean()
	return acc


# parameters of stochastic gradient descent (size of minibatches)
m = Xtrain.shape[0]
b = 20
# plotting real progression of training and test accuracy
# plt.ion()
train_accs = []
test_accs = []
epochs = []

net = pvml.MLP([1600,400, n])
for epoch in range(100):
	net.train(Xtrain, Ytrain, 1e-3, steps=m // b, batch=b)
	if epoch % 5 == 0:
		train_acc = accuracy(net, Xtrain, Ytrain)
		test_acc = accuracy(net, Xtest, Ytest)
		print (epoch, train_acc * 100, test_acc * 100)
		train_accs.append(train_acc * 100)
		test_accs.append(test_acc * 100)
		epochs.append(epoch)
		# plt.clf()
		# plt.plot(epochs, train_accs)
		# plt.plot(epochs, test_accs)
		# plt.ylabel("Accuracy (%)")
		# plt.xlabel("Epochs")
		# plt.legend(["train", "test"])
		# plt.pause(0.01)

# plt.ioff()
# plt.show()
print("Train acc: ", accuracy(net, Xtrain, Ytrain), "Test acc: ", accuracy(net, Xtest, Ytest))
net.save("speech_mlp.npz") #saving parameters



# I used the following code to draw the training curves of different hyoerparameters of the optimizer

# def Training_curve(ax, b, m, ran=100):
# 	train_accs = []
# 	test_accs = []
# 	epochs = []
# 	net = pvml.MLP([1600, 35])
# 	for epoch in range(ran):
# 		net.train(Xtrain, Ytrain, 1e-4, steps=m // b, batch=b)
# 		if epoch % 5 == 0:
# 			train_acc = accuracy(net, Xtrain, Ytrain)
# 			test_acc = accuracy(net, Xtest, Ytest)
# 			print (epoch, train_acc * 100, test_acc * 100)
# 			train_accs.append(train_acc * 100)
# 			test_accs.append(test_acc * 100)
# 			epochs.append(epoch)
# 			ax.plot(epochs, train_accs)
# 			ax.plot(epochs, test_accs)
# 			ax.set_ylabel("Accuracy (%)")
# 			ax.set_xlabel("Epochs")
# 			# ax.legend(["train", "test"])

# m = Xtrain.shape[0]
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize= (15,7))
# b = m
# Training_curve(ax1, b, m)
# b = 1
# Training_curve(ax2, b, m)
# b = 8
# Training_curve(ax3, b, m)
# b = 64
# Training_curve(ax4, b, m)

# plt.show()

plt.plot(epochs, train_accs)
plt.plot(epochs, test_accs)
plt.ylabel("Accuracy (%)")
plt.xlabel("Epochs")
plt.legend(["train", "test"])
