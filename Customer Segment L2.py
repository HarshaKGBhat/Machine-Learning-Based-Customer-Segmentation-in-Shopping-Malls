
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score


seed = 7
print("Libraries updated")


# Set some standard parameters upfront
pd.options.display.float_format = '{:.2f}'.format
sns.set() # Default seaborn look and feel
plt.style.use('ggplot')
print('pandas version ', pd.__version__)


# Load data set containing all the data from csv
df = pd.read_csv('Customer Dataset.csv')
# Describe the data, Shape and how many rows and columns
print('Number of columns in the dataframe: %i' % (df.shape[1]))
print('Number of rows in the dataframe: %i\n' % (df.shape[0]))
print(list(df.columns))
print(df['Segmentation'].value_counts(), '\n')
print( df.head(5), '\n' )
print(df.describe(), '\n')


#not interested columns
del df['ID']
del df['Var_1']
print('Number of columns in the dataframe: %i' % (df.shape[1]))
print('Number of rows in the dataframe: %i\n' % (df.shape[0]))


print(df.isnull().sum())
df.dropna(inplace=True)
print(df.isnull().sum(), '\n')
print('Number of columns in the dataframe: %i' % (df.shape[1]))
print('Number of rows in the dataframe: %i\n' % (df.shape[0]))
print(df['Segmentation'].value_counts(), '\n')


print("Preprocessing Stage, Visualization Level 1 Done")


sns.countplot(df['Gender'],hue=df['Ever_Married'],palette='Set1')
print(pd.crosstab(df['Gender'],df['Ever_Married']))
plt.savefig('Gender vs Ever_Married')
plt.show()

sns.countplot(df['Gender'],hue=df['Profession'],palette='Set1')
print(pd.crosstab(df['Gender'],df['Profession']))
plt.savefig('Gender vs Profession')
plt.show()

# --------- Code Added to improve Accuracy ---------
sns.countplot(df['Graduated'],hue=df['Profession'],palette='Set1')
print(pd.crosstab(df['Graduated'],df['Profession']))
plt.savefig('Graduated vs Profession')
plt.show()

del df['Graduated']

print(df['Profession'].value_counts(), '\n')
index_ref = df[ df['Profession'] == "Artist" ].index
df.drop(index_ref, inplace=True)
index_ref = df[ df['Profession'] == "Homemaker" ].index
df.drop(index_ref, inplace=True)
print(df.shape)

# -----------------------------------------


print(df['Gender'].value_counts(), '\n')
df.Gender = df.Gender.map({'Male':1,'Female':0})
print(df['Gender'].value_counts(), '\n')

print(df['Ever_Married'].value_counts(), '\n')
df.Ever_Married = df.Ever_Married.map({'Yes':1,'No':0})
print(df['Ever_Married'].value_counts(), '\n')


print(df['Profession'].value_counts(), '\n')
df.Profession = df.Profession.map({'Artist':1,'Doctor':2,'Engineer':3,'Entertainment':4,
            'Executive':5,'Healthcare':6,'Homemaker':7,'Lawyer':8,'Marketing':9})
print(df['Profession'].value_counts(), '\n')

print(df['Spending_Score'].value_counts(), '\n')
df.Spending_Score = df.Spending_Score.map({'Low':0,'Average':1,'High':2})
print(df['Spending_Score'].value_counts(), '\n')

print(df['Segmentation'].value_counts(), '\n')
df.Segmentation = df.Segmentation.map({'A':1,'B':2,'C':3,'D':4})
print(df['Segmentation'].value_counts(), '\n')


print("Preprocessing Stage, Visualization Level 2 Done")


X = df.iloc[:,0:8]
#print("X = ",'\n', X)
y = df.iloc[:,-1]
#print('\n', Y)


train_X,test_X,train_y,test_y = train_test_split(
    X,y,test_size=0.15, random_state=seed )


print('\n train_X = \n', train_X)
print('\n test_X = \n', test_X)
print('\n train_y = \n', train_y)
print('\n test_y = \n', test_y)
final_test = test_X.iloc[0:10]
print('\n final_test = \n', final_test)


print("\n Machine Learning Model Build \n")
models=[]
models.append(("logreg",LogisticRegression()))


LR = LogisticRegression()
LR.fit(train_X,train_y)
pred = LR.predict(test_X)
outp = LR.predict(final_test)
print('\n', test_y[:10], '\n')
print("LR Alg Accuracy", accuracy_score(test_y,pred), '\n')
print( "Output Prediction = ", outp, '\n' )


#SVM
SVM_Model = svm.SVC()
SVM_Model.fit(train_X,train_y)
SVM_Prediction = SVM_Model.predict(test_X)
outp = SVM_Model.predict(final_test)
print("SVM Alg Accuracy", int(accuracy_score(test_y,SVM_Prediction) *100), "Percent" )
print( "Output Prediction SVM = ", outp, '\n' )


#KNN
KNN_Model = KNeighborsClassifier()
KNN_Model.fit(train_X,train_y)
KNN_Prediction = KNN_Model.predict(test_X)
outp = KNN_Model.predict(final_test)
print("KNN Alg Accuracy", int(accuracy_score(test_y,KNN_Prediction) *100), "Percent" )
print( "Output Prediction KNN = ", outp, '\n' )


print("Project End")













