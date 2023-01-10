#%%

# Import Packages
import os
import json
import pickle
import datetime
import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from categorize_modules import text_cleaning,lstm_model_creation
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

#%% EDA
#Data Loading

URL = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'

df = pd.read_csv(URL)

#%%
#Data Inspection
df.info()
df.head()
# There are 2 columns : category & text
# 5 categories : 1. tech, 2. business, 3. sport, 4. entertainment, 5. politics

df.duplicated().sum()
#There are 99 duplicated data

#drop the duplicates data
df.drop_duplicates()
#%%
# To clean the data, need to point out the problem
temp = df['text'][0]

#%% 
# Data Cleaning
# 1. Remove tags
# 2. Remove numbers & punctuation
# 3. convert to lower case

for index, temp in enumerate(df['text']):
    df['text'][index] = text_cleaning(temp)

#%%
# Feature Selection
article = df['text']
category = df['category']

#%%
#to calculate no of category
nb_classes = len(np.unique(category))

#%% 
# Data Preprocessing

# A)Tokenizer
num_words = 10000 
oov_token = '<OOV>'  

tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)  # instantiate

tokenizer.fit_on_texts(article)
word_index = tokenizer.word_index
print(dict(list(word_index.items())[0:10]))


article = tokenizer.texts_to_sequences(article)

# (B) Padding
padded_article = pad_sequences(article,maxlen=200,padding='post', truncating='post')

#(C) OneHotEncoder

ohe = OneHotEncoder(sparse=False)
category = ohe.fit_transform(category[::,None])

#To check the category of one hot encoder
#[[0. 0. 0. 1. 0.]] - Sport
#[[1. 0. 0. 0. 0.]] - Business
#[[0. 1. 0. 0. 0.]] - Entertainment
#[[0. 0. 1. 0. 0.]] - Politics
#[[0. 0. 0. 0. 1.]] - Tech


#%%
#Train-Test-Split
X_train,X_test,y_train,y_test = train_test_split(padded_article,category,test_size=0.2,random_state=123)

#%%
#Expand dimension of X_train & X_test
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

#%%
#Model Development
model = lstm_model_creation(num_words,nb_classes)

#Showing the model structure
keras.utils.plot_model(model,show_shapes=True)

#%%
# Tensorboard
LOGS_PATH = os.path.join(os.getcwd(),'Logs',datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

#Tensorboard callbacks
tb = TensorBoard(log_dir = LOGS_PATH)

#%%
#Model Compile
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

EPOCHS = 10
BATCH_SIZE = 64
hist = model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=BATCH_SIZE,epochs=EPOCHS,callbacks=[tb])

#%%
#%% Model Analysis

#To plot the gaph incase tensorboard can't be shown
plt.figure()
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.show()

y_predicted = model.predict(X_test)

#%%
y_predicted = np.argmax(y_predicted, axis=1)
y_test = np.argmax(y_test,axis=1)

print(classification_report(y_test,y_predicted))
cm = confusion_matrix(y_test, y_predicted)
disp = ConfusionMatrixDisplay(cm)
disp.plot()


#%%
#%% Model Saving
#save trained tf model
model.save('model.h5')

#save ohe
with open('ohe.pkl', 'wb') as f:
    pickle.dump(ohe,f)

#tokenizer
with open('tokenizer.json', 'w') as f:
    json.dump(tokenizer.to_json(),f)

