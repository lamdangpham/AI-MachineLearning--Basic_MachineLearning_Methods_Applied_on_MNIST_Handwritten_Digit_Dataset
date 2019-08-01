import numpy as np
import time
import datetime as dt

from sklearn import datasets, svm, metrics
from sklearn.datasets import fetch_mldata


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


#=========================================== 02/ SVM setup with SKLEARN lib
svm_C      = 5
svm_gamma  = 0.001
classifier = svm.SVC(C=svm_C,gamma=svm_gamma)

#=========================================== 03/ Running SVM
start_time = dt.datetime.now()
print('Start learning at {}'.format(str(start_time)))
classifier.fit(seq_x, seq_y)

end_time = dt.datetime.now() 
print('Stop learning {}'.format(str(end_time)))
elapsed_time= end_time - start_time
print('Elapsed learning {}'.format(str(elapsed_time)))


from sklearn.externals import joblib
joblib.dump(classifier, "svm_ker_para.pkl")

#=========================================== 04/ Testing
expected  = seq_y_test
predicted = classifier.predict(seq_x_test)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
cm = metrics.confusion_matrix(expected, predicted)
print("Confusion matrix:\n%s" % cm)
print("Accuracy={}".format(metrics.accuracy_score(expected, predicted)))


