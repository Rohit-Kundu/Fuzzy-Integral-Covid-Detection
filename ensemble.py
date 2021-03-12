import pandas as pd
import numpy as np
from sklearn.metrics import *
import matplotlib.pyplot as plt
import sugeno_integral

def getfile(filename, root="../"):
    file = root+filename+'.csv'
    df = pd.read_csv(file,header=None)
    df = np.asarray(df)

    labels=[]
    for i in range(376):
        labels.append(0)
    for i in range(369):
        labels.append(1)
    labels = np.asarray(labels)
    return df,labels

def predicting(ensemble_prob):
    prediction = np.zeros((ensemble_prob.shape[0],))
    for i in range(ensemble_prob.shape[0]):
        temp = ensemble_prob[i]
        t = np.where(temp == np.max(temp))[0][0]
        prediction[i] = t
    return prediction

def metrics(labels,predictions,classes):
    print("Classification Report:")
    print(classification_report(labels, predictions, target_names = classes,digits = 4))
    matrix = confusion_matrix(labels, predictions)
    print("Confusion matrix:")
    print(matrix)
    print("\nClasswise Accuracy :{}".format(matrix.diagonal()/matrix.sum(axis = 1)))
    print("\nBalanced Accuracy Score: ",balanced_accuracy_score(labels,predictions))

#Sugeno Integral
def ensemble_sugeno(prob1,prob2,prob3,prob4):
    num_classes = prob1.shape[1]
    Y = np.zeros(prob1.shape,dtype=float)
    for samples in range(prob1.shape[0]):
        for classes in range(prob1.shape[1]):
            X = np.array([prob1[samples][classes], prob2[samples][classes], prob3[samples][classes], prob4[samples][classes] ])
            measure = np.array([1.5, 1.5, 0.01, 1.2])
            X_agg = integrals.sugeno_fuzzy_integral_generalized(X,measure)
            Y[samples][classes] = X_agg

    sugeno_pred = predicting(Y)

    correct = np.where(sugeno_pred == labels)[0].shape[0]
    total = labels.shape[0]

    print("Accuracy = ",correct/total)
    classes = ['COVID','Non-COVID']
    metrics(sugeno_pred,labels,classes)
