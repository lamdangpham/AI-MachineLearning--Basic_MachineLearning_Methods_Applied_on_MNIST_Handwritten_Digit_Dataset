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
from scipy.stats import uniform as sp_uniform
C_svm     = sp_uniform(scale=10)
gamma_svm = sp_uniform(scale=1)
parameters = {'kernel': ['rbf'],
              'C': C_svm,
              'gamma': gamma_svm
             }

from sklearn.model_selection import RandomizedSearchCV
n_iter_search = 8
svm_model = svm.SVC()
classifier = RandomizedSearchCV(estimator=svm_model,
                                param_distributions=parameters,
                                n_iter=n_iter_search,
                                cv=3,
                                n_jobs=1,
                                verbose=2
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
goal_params     = classifier.best_params_

from sklearn.externals import joblib
joblib.dump(goal_classifier, "svm_ker_rand_search_para.pkl")

range_C     = classifier.cv_results_['param_C']
range_gamma = classifier.cv_results_['param_gamma']
scores      = classifier.cv_results_['mean_test_score']

#=========================================== 04/ Testing
expected  = seq_y_test
predicted = goal_classifier.predict(seq_x_test)

print("\n Classification report for classifier: %s \n"
      % (goal_classifier))
print("\n %s \n"
      % (metrics.classification_report(expected, predicted)))
print("\n")
cm = metrics.confusion_matrix(expected, predicted)
print("Confusion matrix: \n%s" % cm)
print("\n")
print("Accuracy: {}".format(metrics.accuracy_score(expected, predicted)))
