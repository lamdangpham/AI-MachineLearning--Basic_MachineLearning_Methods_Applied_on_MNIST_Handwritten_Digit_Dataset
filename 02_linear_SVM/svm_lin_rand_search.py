import matplotlib.pyplot as plt
import numpy as np
import time
import datetime as dt

from sklearn import datasets, svm, metrics
from sklearn.datasets import fetch_mldata
#from mnist_helpers import *


#=========================================== 01/ Import Input Data
print("\n ==================================================================== IMPORT DATA...")
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

seq_x = mnist.train.images      #55000:784
seq_y = mnist.train.labels      #55000:10
seq_x_test = mnist.test.images  #10000:784
seq_y_test = mnist.test.labels  #10000:10

#normalize input images energy from [0:255] into [0:1]
seq_x = seq_x/255.0
seq_x_test = seq_x_test/255.0

#tranfer input expected from one-hot format (2-D matrix) into one column vector
seq_y      = np.argmax(seq_y, axis=1)
seq_y_test = np.argmax(seq_y_test, axis=1)


#=========================================== 02/ SVM setup with SKLEARN lib
from scipy.stats import uniform as sp_uniform
C_svm      = sp_uniform(scale=10)
parameters = {'C':C_svm}  #random search only on C_svm para

svm_model = svm.LinearSVC()
n_iter_search = 2

from sklearn.model_selection import RandomizedSearchCV
classifier = RandomizedSearchCV(estimator           = svm_model,
                                param_distributions = parameters,
                                n_iter              = n_iter_search, 
                                cv                  = 3,
                                n_jobs              = 1,
                                verbose             = 2
                               )

#=========================================== 03/ Running SVM
start_time = dt.datetime.now()
print('Start learning at {}'.format(str(start_time)))
classifier.fit(seq_x, seq_y)

end_time = dt.datetime.now() 
print('Stop learning {}'.format(str(end_time)))
elapsed_time= end_time - start_time
print('Elapsed learning {}'.format(str(elapsed_time)))

goal_classifier = classifier.best_estimator_
goal_paras      = classifier.best_params_


range_C   = classifier.cv_results_['param_C']
ave_score = classifier.cv_results_['mean_test_score']
print(range_C)
print(ave_score)
#=========================================== 04/ Testing
expected  = seq_y_test
predicted = goal_classifier.predict(seq_x_test)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
cm = metrics.confusion_matrix(expected, predicted)
print("Confusion matrix:\n%s" % cm)
print("Accuracy={}".format(metrics.accuracy_score(expected, predicted)))


