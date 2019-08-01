from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle

#=========================================== 01/ Import Input Data
print("\n ==================================================================== IMPORT DATA...")
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

seq_x = mnist.train.images      #55000:784
seq_y = mnist.train.labels      #55000:10
seq_x_test = mnist.test.images  #10000:784
seq_y_test = mnist.test.labels  #10000:10

#normalize input images
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
seq_x      = scaler.fit_transform(seq_x)
seq_x_test = scaler.fit_transform(seq_x_test)

#tranfer input expected from one-hot format (2-D matrix) into one column vector
seq_y      = np.argmax(seq_y, axis=1)
seq_y_test = np.argmax(seq_y_test, axis=1)


print("\n ==================================================================== TRAINING...")
classifier = RandomForestClassifier(n_estimators=100, n_jobs=10,)
classifier.fit(seq_x,seq_y)

print("\n ==================================================================== STORING PARA...")
with open('RF_para','wb') as f:
	pickle.dump(classifier, f)

print("\n ==================================================================== LOAD PARA & VERIFYING...")
pickle_in = open('RF_para','rb')
goal_classifier = pickle.load(pickle_in)

acc = goal_classifier.score(seq_x_test, seq_y_test)
print('RFC Score: ',acc)
