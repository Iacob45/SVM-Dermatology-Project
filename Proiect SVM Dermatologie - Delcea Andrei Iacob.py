import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm,linear_model
from sklearn import metrics
from sklearn import model_selection



database = pandas.read_csv(r"\dermatology.data",na_values='?' , delimiter=',')
database = database.fillna(30)
array = database.values

data = array[:, :34]
tags = array[:,34]
data=data.astype('int')
tags=tags.astype('int')




validation_size = 0.3
seed = 69
X, X_test, Y, Y_test = model_selection.train_test_split(data, tags, test_size=validation_size, random_state=seed)


b=-5

for i in range(7):
    clf = svm.SVC(kernel='linear',C=2**b,gamma=1)
    clf.fit(X, Y)
    predictie = clf.predict(X_test)
    print("Precizie pentru cost= 2^", b, ": ", metrics.accuracy_score(Y_test, predictie))
    b = b + 2




