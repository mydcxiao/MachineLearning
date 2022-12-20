from xml.dom import ValidationErr
import numpy as np
import json
import collections
import matplotlib.pyplot as plt


def data_processing(data):
	train_set, valid_set, test_set = data['train'], data['valid'], data['test']
	Xtrain = train_set[0]
	ytrain = train_set[1]
	Xval = valid_set[0]
	yval = valid_set[1]
	Xtest = test_set[0]
	ytest = test_set[1]

	Xtrain = np.array(Xtrain)
	Xval = np.array(Xval)
	Xtest = np.array(Xtest)

	ytrain = np.array(ytrain)
	yval = np.array(yval)
	ytest = np.array(ytest)
	
	return Xtrain, ytrain, Xval, yval, Xtest, ytest


def data_processing_with_transformation(data, do_minmax_scaling=True, do_normalization=False):
	train_set, valid_set, test_set = data['train'], data['valid'], data['test']
	Xtrain = train_set[0]
	ytrain = train_set[1]
	Xval = valid_set[0]
	yval = valid_set[1]
	Xtest = test_set[0]
	ytest = test_set[1]

	Xtrain = np.array(Xtrain)
	Xval = np.array(Xval)
	Xtest = np.array(Xtest)

	ytrain = np.array(ytrain)
	yval = np.array(yval)
	ytest = np.array(ytest)
	
	# We load data from json here and turn the data into numpy array
	# You can further perform data transformation on Xtrain, Xval, Xtest

	# Min-Max scaling
	if do_minmax_scaling:
		#####################################################
		#				 YOUR CODE HERE					    #
		#####################################################
		m = np.shape(Xtrain)[0]
		n = np.shape(Xtrain)[1]
		slope = np.empty(n)
		offset = np.empty(n)
		for i in range(n):
			max_v = max(Xtrain[:, i])
			min_v = min(Xtrain[:, i])
			slope[i] = 1 / (max_v - min_v)
			offset[i] = - min_v / (max_v - min_v)
			for j in range(m):
				Xtrain[j, i] = Xtrain[j, i] * slope[i] + offset[i]
		for i in range(np.shape(Xval)[1]):
			for j in range(np.shape(Xval)[0]):
				Xval[j, i] = Xval[j, i] * slope[i] + offset[i]
		for i in range(np.shape(Xtest)[1]):
			for j in range(np.shape(Xtest)[0]):
				Xtest[j, i] = Xtest[j, i] * slope[i] + offset[i]
			
	# Normalization
	def normalization(x):
		#####################################################
		#				 YOUR CODE HERE					    #
		#####################################################
		m = np.shape(x)[0]
		n = np.shape(x)[1]
		for i in range(m):
			norm = np.linalg.norm(x[i]) 
			for j in range(n):
				x[i, j] = x[i, j] / norm if norm != 0 else 0
		return x
	
	if do_normalization:
		Xtrain = normalization(Xtrain)
		Xval = normalization(Xval)
		Xtest = normalization(Xtest)

	return Xtrain, ytrain, Xval, yval, Xtest, ytest


def compute_l2_distances(Xtrain, X):
	"""
	Compute the distance between each test point in X and each training point
	in Xtrain.
	Inputs:
	- Xtrain: A numpy array of shape (num_train, D) containing training data
	- X: A numpy array of shape (num_test, D) containing test data.
	Returns:
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  is the Euclidean distance between the ith test point and the jth training
	  point.
	"""
	#####################################################
	#				 YOUR CODE HERE					    #
	#####################################################
	m = np.shape(Xtrain)[0]
	n = np.shape(X)[0]
	dists = np.empty((n, m))
	for i in range(n):
		for j in range(m):
			dists[i, j] = np.linalg.norm(Xtrain[j] - X[i])
	return dists


def compute_cosine_distances(Xtrain, X):
	"""
	Compute the distance between each test point in X and each training point
	in Xtrain.
	Inputs:
	- Xtrain: A numpy array of shape (num_train, D) containing training data
	- X: A numpy array of shape (num_test, D) containing test data.
	Returns:
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  is the Cosine distance between the ith test point and the jth training
	  point.
	"""
	#####################################################
	#				 YOUR CODE HERE					    #
	#####################################################
	m = np.shape(Xtrain)[0]
	n = np.shape(X)[0]
	dists = np.empty((n, m))
	for i in range(n):
		for j in range(m):
			dists[i, j] = 1 if np.linalg.norm(Xtrain[j]) == 0 or np.linalg.norm(X[i]) == 0 else\
				1 - np.dot(Xtrain[j], X[i]) / (np.linalg.norm(Xtrain[j]) * np.linalg.norm(X[i])) 
	return dists


def predict_labels(k, ytrain, dists):
	"""
	Given a matrix of distances between test points and training points,
	predict a label for each test point.
	Inputs:
	- k: The number of nearest neighbors used for prediction.
	- ytrain: A numpy array of shape (num_train,) where ytrain[i] is the label
	  of the ith training point.
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  gives the distance betwen the ith test point and the jth training point.
	Returns:
	- ypred: A numpy array of shape (num_test,) containing predicted labels for the
	  test data, where y[i] is the predicted label for the test point X[i].
	"""
	#####################################################
	#				 YOUR CODE HERE					    #
	#####################################################
	m = np.shape(dists)[0]
	n = np.shape(dists)[1]
	index_arr = np.argpartition(dists, k)
	ypred = np.empty(m)
	for i in range(m):
		dict = {}
		for j in range(k):
			key = ytrain[index_arr[i, j]]
			dict[key] = dict.get(key, 0) + 1
		max_v = max(dict.values())
		max_k = np.array([k for k in dict if dict[k] == max_v])
		ypred[i] = min(max_k)
	return ypred


def compute_error_rate(y, ypred):
	"""
	Compute the error rate of prediction based on the true labels.
	Inputs:
	- y: A numpy array with of shape (num_test,) where y[i] is the true label
	  of the ith test point.
	- ypred: A numpy array with of shape (num_test,) where ypred[i] is the
	  prediction of the ith test point.
	Returns:
	- err: The error rate of prediction (scalar).
	"""
	#####################################################
	#				 YOUR CODE HERE					    #
	#####################################################
	num = np.shape(y)[0]
	e_num = 0
	for i in range(num):
		if ypred[i] != y[i]:
			e_num += 1
	err = e_num / num
	return err


def find_best_k(K, ytrain, dists, yval):
	"""
	Find best k according to validation error rate.
	Inputs:
	- K: A list of ks.
	- ytrain: A numpy array of shape (num_train,) where ytrain[i] is the label
	  of the ith training point.
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  is the distance between the ith test point and the jth training
	  point.
	- yval: A numpy array with of shape (num_val,) where y[i] is the true label
	  of the ith validation point.
	Returns:
	- best_k: The k with the lowest error rate.
	- validation_error: A list of error rate of different ks in K.
	- best_err: The lowest error rate we get from all ks in K.
	"""
	#####################################################
	#				 YOUR CODE HERE					    #
	#####################################################
	validation_error = []
	best_err = None
	best_k = None
	for k in K:
		ypred = predict_labels(k, ytrain, dists)
		error = compute_error_rate(yval, ypred)
		if best_err is None or error < best_err:
			best_err = error
			best_k = k
		validation_error.append(error)
	return best_k, validation_error, best_err


def main():
	input_file = 'disease.json'
	output_file = 'knn_output.txt'

	#==================Problem Set 1.1=======================

	with open(input_file) as json_data:
		data = json.load(json_data)

	# Compute distance matrix
	Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing(data)

	dists = compute_l2_distances(Xtrain, Xval)

	# Compute validation accuracy when k=4
	k = 4
	ypred = predict_labels(k, ytrain, dists)
	err = compute_error_rate(yval, ypred)
	print("The validation error rate is", err, "in Problem Set 1.1")
	print()

	#==================Problem Set 1.2=======================

	# Compute distance matrix
	Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing_with_transformation(data, do_minmax_scaling=False, do_normalization=True)

	dists = compute_l2_distances(Xtrain, Xval)

	# Compute validation accuracy when k=4
	k = 4
	ypred = predict_labels(k, ytrain, dists)
	err = compute_error_rate(yval, ypred)
	print("The validation error rate is", err, "in Problem Set 1.2 when using normalization")
	print()

	# Compute distance matrix
	Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing_with_transformation(data, do_minmax_scaling=True, do_normalization=False)

	dists = compute_l2_distances(Xtrain, Xval)

	# Compute validation accuracy when k=4
	k = 4
	ypred = predict_labels(k, ytrain, dists)
	err = compute_error_rate(yval, ypred)
	print("The validation error rate is", err, "in Problem Set 1.2 when using minmax_scaling")
	print()
	
	#==================Problem Set 1.3=======================

	# Compute distance matrix
	Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing(data)
	dists = compute_cosine_distances(Xtrain, Xval)

	# Compute validation accuracy when k=4
	k = 4
	ypred = predict_labels(k, ytrain, dists)
	err = compute_error_rate(yval, ypred)
	print("The validation error rate is", err, "in Problem Set 1.3, which use cosine distance")
	print()

	#==================Problem Set 1.4=======================
	# Compute distance matrix
	Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing(data)

	#======performance of different k in training set=====
	K = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18]
	#####################################################
	#				 YOUR CODE HERE					    #
	#####################################################
	#======plot the error rate of training set for each k and save it===
	dists = compute_l2_distances(Xtrain, Xtrain)
	best_k, training_error, best_err = find_best_k(K, ytrain, dists, ytrain)
	plt.figure()
	plt.plot(K, training_error, label="training error rate")
	plt.xlim((0,20))
	plt.xticks(np.arange(0,20,2))
	plt.tick_params(axis='x', direction='in')
	plt.tick_params(axis='y', direction='in')
	plt.xlabel("hyper-parameter k")
	plt.ylabel("error rate")
	plt.title("Training error rate over k")
	plt.legend()
	plt.savefig("T_err_k.png", bbox_inches='tight')
	
	#==========select the best k by using validation set==============
	dists = compute_l2_distances(Xtrain, Xval)
	best_k, validation_error, best_err = find_best_k(K, ytrain, dists, yval)
	plt.figure()
	plt.plot(K, validation_error, label="validation error rate")
	plt.xticks(np.arange(0,20,2))
	plt.tick_params(axis='x', direction='in')
	plt.tick_params(axis='y', direction='in')
	plt.xlabel("hyper-parameter k")
	plt.ylabel("error rate")
	plt.title("Validation error rate over k")
	plt.legend()
	plt.savefig("V_err_k.png", bbox_inches='tight')

	#===============test the performance with your best k=============
	dists = compute_l2_distances(Xtrain, Xtest)
	ypred = predict_labels(best_k, ytrain, dists)
	test_err = compute_error_rate(ytest, ypred)
	print("In Problem Set 1.4, we use the best k = ", best_k, "with the best validation error rate", best_err)
	print("Using the best k, the final test error rate is", test_err)
	#====================write your results to file===================
	f=open(output_file, 'w')
	f.write("Training error over k:\n")
	for i in range(len(K)):
		f.write('%d %.3f' % (K[i], training_error[i])+'\n')
	f.write("Validation error over k:\n")
	for i in range(len(K)):
		f.write('%d %.3f' % (K[i], validation_error[i])+'\n')
	f.write('%s %.3f' % ('test', test_err))
	f.close()

if __name__ == "__main__":
	main()
