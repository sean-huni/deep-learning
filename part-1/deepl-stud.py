##############@@@@@@@@####################
## STEP 1: DATA IMPORT & PRE-PROCESSING
##############@@@@@@@@####################
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

############################
# 1.2: Import the dataset
############################
# Importing the dataset
dt = pd.read_csv('stud-por-preformat.csv')

###########################
# 1.3: Data-Type Correction
###########################
# Mapping the dataset to their correct datatypes.
dt = dt.astype({'school':'category', 'sex':'category', 'age':'float64', 'address':'category', 'famsize':'category', 'Pstatus':'category', 'Medu':'float64', 'Fedu':'float64', 'Mjob':'category', 'Fjob':'category', 'reason':'category', 'guardian':'category', 'traveltime':'float64', 'studytime':'float64', 'failures':'float64', 'schoolsup':'bool', 'famsup':'bool', 'paid':'bool', 'activities':'bool', 'nursery':'bool', 'higher':'bool', 'internet':'bool', 'romantic':'bool', 'famrel':'float64', 'freetime':'float64', 'goout':'float64', 'Dalc':'float64', 'Walc':'float64', 'health':'float64', 'absences':'float64', 'G1':'float64', 'G2':'float64', 'G3':'float64'}) 

# Shuffle the order of the records to clear out any biases.
from sklearn.utils import shuffle
dt = shuffle(dt)

# Lets do some integrity datatype checks & prints to the console
print(dt['age'].dtypes)
print(dt['school'].dtypes)
print(dt.head(5)) 

###########################
# 1.4: Extracting independant/dependant variables
###########################
# Insert a Final-Grade decision binary field/column. (Gfinal Range 0-20: True if x >= 10, otherwise false) 
dt.insert(33, 'Gfinal', dt.iloc[:, 32].values >= 10)      
print(dt['Gfinal'].dtypes)

# Independant Variables
X = dt.iloc[:, 0:32]

# Dependant Variable
y = dt.iloc[:, 33]

###########################
# 1.5: Encoding Categorical Fields
###########################
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
cat_fields_no = [0, 1, 3, 4, 5, 8, 9, 10, 11] # Column/Field-No.s with categoriacal data.

lex1 = LabelEncoder()
X['school'] = lex1.fit_transform(X['school'])
lex2 = LabelEncoder()
X['sex'] = lex2.fit_transform(X['sex'])
lex3 = LabelEncoder()
X['address'] = lex3.fit_transform(X['address'])
lex4 = LabelEncoder()
X['famsize'] = lex4.fit_transform(X['famsize'])
lex5 = LabelEncoder()
X['Pstatus'] = lex5.fit_transform(X['Pstatus'])
lex6 = LabelEncoder()
X['Mjob'] = lex6.fit_transform(X['Mjob'])
lex7 = LabelEncoder()
X['Fjob'] = lex7.fit_transform(X['Fjob'])
lex8 = LabelEncoder()
X['reason'] = lex8.fit_transform(X['reason'])
lex9 = LabelEncoder()
X['guardian'] = lex9.fit_transform(X['guardian'])

lex10 = LabelEncoder()
X['schoolsup'] = lex10.fit_transform(X['schoolsup'])
lex11 = LabelEncoder()
X['famsup'] = lex11.fit_transform(X['famsup'])
lex12 = LabelEncoder()
X['paid'] = lex12.fit_transform(X['paid'])
lex13 = LabelEncoder()
X['activities'] = lex13.fit_transform(X['activities'])
lex14 = LabelEncoder()
X['nursery'] = lex14.fit_transform(X['nursery'])
lex15 = LabelEncoder()
X['higher'] = lex15.fit_transform(X['higher'])
lex16 = LabelEncoder()
X['internet'] = lex16.fit_transform(X['guardian'])
lex17 = LabelEncoder()
X['romantic'] = lex17.fit_transform(X['romantic'])

ohe = OneHotEncoder(categorical_features = cat_fields_no)
X = ohe.fit_transform(X).toarray()

# Already Done
# ley = LabelEncoder()
# y = ley.fit_transform(y)
  
###########################
# 1.6: Split Training & Test Sets
###########################
# Splitting the dataset into Training & Test Sets.
from sklearn.model_selection import train_test_split as tts
X_train, X_test, Y_train, Y_test = tts(X, y, test_size = 0.2, random_state = 0)

###########################
# 1.7: Feature Scaling
###########################
# Applying Feature Scaling >>> Around the Zero-Mean-Variance
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


##############@@@@@@@@####################
## STEP 2: CREATE & FIT THE DATA TO THE ANN
##############@@@@@@@@####################

# Importing the Keras Labraries & Packages
import keras
from keras.models import Sequential
from keras.layers import Dense

###########################
# 2.1: Initialise the ANN
###########################
classifier = Sequential()

# Input & 1st Hidden Layer
classifier.add(Dense(activation="relu", input_dim=50, units=26, kernel_initializer="uniform"))

# 2nd Hidden Layer
classifier.add(Dense(activation="relu", units=26, kernel_initializer="uniform"))

# 3rd Hidden Layer
classifier.add(Dense(activation="relu", units=26, kernel_initializer="uniform"))

# 4th Hidden Layer
classifier.add(Dense(activation="relu", units=26, kernel_initializer="uniform"))

# 5th Hidden Layer
classifier.add(Dense(activation="relu", units=26, kernel_initializer="uniform"))

# 6th Hidden Layer
classifier.add(Dense(activation="relu", units=26, kernel_initializer="uniform"))

# 7th Hidden Layer
classifier.add(Dense(activation="relu", units=20, kernel_initializer="uniform"))

# 8th Hidden Layer
classifier.add(Dense(activation="relu", units=18, kernel_initializer="uniform"))

# 9th Hidden Layer
classifier.add(Dense(activation="relu", units=16, kernel_initializer="uniform"))

# 10th Hidden Layer
classifier.add(Dense(activation="relu", units=12, kernel_initializer="uniform"))

# Output Layer
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

###########################
# 2.2: Compile the ANN
###########################
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

###########################
# 2.3: Train the ANN
###########################
h_cb = classifier.fit(X_train, Y_train, batch_size = 10, nb_epoch = 1000)

##############@@@@@@@@####################
## STEP 3: ANN MODEL EVALUATION & PREDICTIONS
##############@@@@@@@@####################

###########################
# 3.1: Making predictions
###########################
# Use the classfisier to make predictions on the test-set.
y_pred = classifier.predict(X_test)
y_pred = (y_pred >= 0.5)

###########################
# 3.2: Model Evaluation
###########################
# Evaluate the predictions using the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm = confusion_matrix(Y_test, y_pred)

acc = accuracy_score(Y_test, y_pred) * 100.00

print (f'Accuracy : {acc}')


##############@@@@@@@@####################
# 4: Visualise the Training Loss & Accuracy
##############@@@@@@@@####################
# Keys to plot
print(h_cb.history.keys()) 

###########################
# 4.1: Plot Training Accuracy
###########################
plt.plot(h_cb.history['acc'], color='blue')
plt.title('Training Accuracy vs Epoch')
plt.ylabel('Training Accuracy')
plt.xlabel('Epoch')
plt.grid(True)
plt.legend(['Training-accuracy'], loc='best')
plt.show()

###########################
# 4.2: Plot Training Loss
###########################
plt.plot(h_cb.history['loss'], color='g')
plt.title('Training Loss vs Epoch')
plt.ylabel('Training Loss')
plt.xlabel('Epoch')
plt.legend(['Training-loss'], loc='best')
plt.grid(True)
plt.show()

