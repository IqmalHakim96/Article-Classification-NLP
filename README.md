# Article-Classification-NLP
 
 ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
 ![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)

Natural Learning Process (NLP) used to analysed the text data. There is a lot applications using NLP around us nowadays. In this analysis, the NLP was used with deep learning model which is LSTM neural network approach to classified or categorize the article.

OBJECTIVE - Create classifier model to categorize the article into different categories using deep learning.

### Methodology

1. Model training = deep learning
2. Model = Sequential, LSTM
3. Module = Tensorflow & Sklearn

### About Dataset

There are 2225 text data entries with 5 different categories (Tech,Sport,Business,Politics & Entertainment)

The text data have 99 duplicated data and need to cleared before seperating the dataset

Before start the training of the data, The text data need to be clean from any anomilities, numbers and be in lower case alphabet. 
For category data, one hot encoder was used to format that can be traine in deep learning

### Deep Learning with LSTM
A sequential model was created with 2 layer of LSTM and 1 Dense layer:
![model_sequential](https://user-images.githubusercontent.com/105650253/211496015-c02ec8e9-b32d-4864-95cf-d5cf9a70690b.PNG)

![model_architecture](https://user-images.githubusercontent.com/105650253/211496096-d29b4c55-aba7-4578-9140-3cfb86531d08.PNG)

The data were trained with 10 epochs

![Tensorboard_Graph](https://user-images.githubusercontent.com/105650253/211496269-26052aef-cd7e-485a-b3ba-7d17707c46b7.PNG)

The classification report, confusion matrix, accuracy were shown in figure below
![report](https://user-images.githubusercontent.com/105650253/211496475-7366f39b-5878-429c-916f-127ac2623d13.PNG)

The f1 score is 0.84 or 84%

### Discussion
This model is good to train the text data however there is some improvement that can be applied to ensure the result of F1-Score and Accuracy can be exceed to 90%:
 1. Number of epochs can be add up with parameters of dropout to ensure the training model are not overfitting
 2. The LSTM neural network model can be improve by adding the number of nodes, and Dense layers can include it in between the layers.

### Acknowledgement
In this analysis, the dataset was used from : https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv

