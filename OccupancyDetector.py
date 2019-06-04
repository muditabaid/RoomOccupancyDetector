
# coding: utf-8

# In[1]:


import pandas as pd, numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from pandas import read_csv
from matplotlib import pyplot
from pandas import concat
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_auc_score,roc_curve
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from pandas.plotting import scatter_matrix
import pickle


# In[2]:


# load all data
data1 = read_csv('datatest.txt', header=0, index_col=1, parse_dates=True, squeeze=True)
data2 = read_csv('datatraining.txt', header=0, index_col=1, parse_dates=True, squeeze=True)
data3 = read_csv('datatest2.txt', header=0, index_col=1, parse_dates=True, squeeze=True)


# In[3]:


# vertically stack and maintain temporal order
data = concat([data1, data2, data3])

# save aggregated dataset
data.to_csv('combined.csv')

# load the dataset
data = read_csv('combined.csv')

# print(data)
# print(data.shape)


# In[4]:


data.head()


# In[5]:


# drop the feature sno and date since it is just index of the data.No dependence with final result.
data.drop(['sno','date'], axis=1, inplace=True)
data.head()


# In[6]:


print('DATA VISUALIZATION')


# In[7]:


data.corr()


# In[8]:


#features humidityRatio and humidity has hight correlation. 
# high correlation 
print('Correlation Matrix')
plt.figure(figsize=(8,8))
plt.matshow(data.corr(),fignum=1)
plt.xticks(range(len(data.columns)), data.columns)
plt.yticks(range(len(data.columns)), data.columns)
plt.colorbar()
plt.show()


# In[9]:


#Pair-wise Relationships between attributes
print('Pair-wise Relationships')
scatter_matrix(data,figsize=(10,10))


# In[10]:


#Features 'HumidityRatio' and 'Humidity' are fully linear dependent.Therefore drop one of the feature.
data.drop(['HumidityRatio'], axis=1, inplace=True)
data.head()


# In[11]:


print('Accuracy Graph of Simple Logistic Regression Model on each environment measure in isolation')
values = data.values
features = [0, 1, 2, 3]
for f in features:
	# split data into inputs and outputs
	X1, y1 = values[:, f].reshape((len(values), 1)), values[:, -1]
	# split the dataset
	train_X, test_X, train_y, test_y = train_test_split(X1, y1, test_size=0.3, random_state=1)
	# define the model
	model = LogisticRegression()
	# fit the model on the training set
	model.fit(train_X, train_y)
	# predict the test set
	y_hat = model.predict(test_X)
	# evaluate model skill
	score = accuracy_score(test_y, y_hat)
	plt.hlines(y=data.columns[f], xmin=0, xmax=score*100)


# In[12]:


#We can see that only the “Light” feature is required in order to achieve 99% accuracy
#Very likely that the office rooms in which the environmental variables 
#were recorded had a light sensor that turned internal lights on
#when the room was occupied and off otherwise.
#We decided to remove this feature assuming there is no light sensor in room to generalize the model.
data.drop(['Light'], axis=1,inplace=True)
data.head()


# In[13]:


values = data.values


# split data into inputs and outputs
X, y = values[:, :-1], values[:, -1]

# split the dataset
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.3, random_state=1)



# In[14]:


def LogisticReg():
    # Dump the "TRAINED Logistic Regression" classifier with Pickle
    logistic_reg_pkl_filename = 'logistic_reg_classifier.pkl'
    #     Open the file to save as pkl file
#     logistic_reg_model_pkl = open(logistic_reg_pkl_filename, 'wb') 
#     model = LogisticRegression()
       # fit the model on the training set
#     model.fit(trainX, trainy)
    #dump model
#     pickle.dump(model,logistic_reg_model_pkl)
    
     # Loading the saved naive bayes model pickle
    logistic_reg_model_pkl = open(logistic_reg_pkl_filename, 'rb')
    model = pickle.load(logistic_reg_model_pkl)

    # predict the test set
    predy = model.predict(testX)
    
    #save output in csv file
    logreg_out =pd.DataFrame({'OccupancyLogReg': predy})


    # evaluate model skil
    print('Accuracy of logistic regression classifier on test set: {:.4f}\n'.format(accuracy_score(testy,predy)))
    
    print('Confusion matrix:')
    print(confusion_matrix(testy,predy))

    print('\nReport:')
    print(classification_report(testy,predy))
    
    print('ROC_Curve')
    logit_roc_auc = roc_auc_score(testy, model.predict(testX))
    fpr, tpr, thresholds = roc_curve(testy, model.predict_proba(testX)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()
    # Close the pickle instances
    logistic_reg_model_pkl.close()
    return logreg_out


# In[15]:


logreg_out=LogisticReg()


# In[16]:


def NaiveBayes():
    # Dump the "TRAINED NAIVE BAYES" classifier with Pickle
    naive_bayes_pkl_filename = 'naive_bayes_classifier.pkl'
    #     Open the file to save as pkl file
#     naive_bayes_model_pkl = open(naive_bayes_pkl_filename, 'wb')
#     model = GaussianNB()
#     model.fit(trainX,trainy)
    # Dump model
#     pickle.dump(model, naive_bayes_model_pkl)


    # Loading the saved naive bayes model pickle
    naive_bayes_model_pkl = open(naive_bayes_pkl_filename, 'rb')
    model = pickle.load(naive_bayes_model_pkl)
    
    #predict Occupancy
    predy= model.predict(testX)
    
    #save output in a file
    gaussian_out =pd.DataFrame({'OccupancyNaiveBayes': predy})
    
    print('Accuracy of Naive Bayes classifier on test set: {:.4f}\n'.format(accuracy_score(testy,predy)))
    
    print('Confusion matrix:')
    print(confusion_matrix(testy,predy))
    
    print('\nReport:')
    print(classification_report(testy,predy))
    
    print('ROC_Curve')
    logit_roc_auc = roc_auc_score(testy, model.predict(testX))
    fpr, tpr, thresholds = roc_curve(testy, model.predict_proba(testX)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='Naive Bayes (area = %0.3f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()
    # Close the pickle instances
    naive_bayes_model_pkl.close()
    return gaussian_out


# In[17]:


gaussian_out=NaiveBayes()


# In[18]:


def DecisionTree():
    # Dump the "TRAINED DECISION TREE" classifier with Pickle
    decision_tree_pkl_filename = 'decision_tree_classifier.pkl'
#     Open the file to save as pkl file
#     decision_tree_model_pkl = open(decision_tree_pkl_filename, 'wb')
#     tree = DecisionTreeClassifier()
#     tree.fit(trainX, trainy)
#     pickle.dump(tree, decision_tree_model_pkl)

    # Loading the saved decision tree model pickle
    decision_tree_model_pkl = open(decision_tree_pkl_filename, 'rb')
    model = pickle.load(decision_tree_model_pkl)
    
    #predict occupancy 
    predy = model.predict(testX)
    
    # save output in csv file
    dectree_out =pd.DataFrame({'Occ_DecisionTree': predy})
    
    print('Accuracy of Decision Tree classifier on test set: {:.4f}\n'.format(accuracy_score(testy,predy)))
    from sklearn.metrics import classification_report, confusion_matrix  
    
    print('Confusion matrix:')
    print(confusion_matrix(testy, predy))  
    
    print('\nReport:')
    print(classification_report(testy, predy))  
    
    print('ROC_Curve')
    logit_roc_auc = roc_auc_score(testy, model.predict(testX))
    fpr, tpr, thresholds = roc_curve(testy, model.predict_proba(testX)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='Decision Tree (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()
    # Close the pickle instances
    decision_tree_model_pkl.close()
    return dectree_out


# In[19]:


dectree_out=DecisionTree()


# In[20]:


def RandomForest():
   # Dump the "TRAINED RANDOM FOREST" classifier with Pickle
   random_forest_pkl_filename = 'random_forest_classifier.pkl'

#      Open the file to save as pkl file
#     random_forest_model_pkl = open(random_forest_pkl_filename, 'wb')
#     Create the model with 1500 trees
#     model = RandomForestClassifier(n_estimators=1500, 
#                                    bootstrap = True,
#                                    max_features = 'sqrt')
#      Fit on training data
#     model.fit(trainX, trainy)
    # Dump model
#     pickle.dump(model, random_forest_model_pkl)

   
   # Loading the saved random forest model pickle
   random_forest_model_pkl = open(random_forest_pkl_filename, 'rb')
   model= pickle.load(random_forest_model_pkl)
   
   # predict occupancy
   rf_predictions = model.predict(testX)
   
   # save output in csv file
   RandomForest_out =pd.DataFrame({'Occ_RandomForest': rf_predictions})
   
   # predict occupancy probabilities
   rf_probs = model.predict_proba(testX)[:, 1]

   print('Accuracy of Random Forest classifier on test set: {:.4f}\n'.format(accuracy_score(testy, rf_predictions)))
   
   print('Confusion matrix:')
   print(confusion_matrix(testy, rf_predictions))  
   
   print('\nReport:')
   print(classification_report(testy, rf_predictions))  
   
   print('ROC_Curve')
   logit_roc_auc = roc_auc_score(testy, model.predict(testX))
   fpr, tpr, thresholds = roc_curve(testy, model.predict_proba(testX)[:,1])
   pyplot.figure()
   pyplot.plot(fpr, tpr, label='Random Forest (area = %0.2f)' % logit_roc_auc)
   pyplot.plot([0, 1], [0, 1],'r--')
   pyplot.xlim([0.0, 1.0])
   pyplot.ylim([0.0, 1.05])
   pyplot.xlabel('False Positive Rate')
   pyplot.ylabel('True Positive Rate')
   pyplot.title('Receiver operating characteristic')
   pyplot.legend(loc="lower right")
   pyplot.savefig('Log_ROC')
   pyplot.show()
   # Close the pickle instances
   random_forest_model_pkl.close()
   return RandomForest_out


# In[21]:


RandomForest_out=RandomForest()


# In[22]:


submission = pd.concat([logreg_out,gaussian_out,dectree_out,RandomForest_out ], axis=1)
submission.to_csv('output2.csv', index=False)


# In[ ]:




