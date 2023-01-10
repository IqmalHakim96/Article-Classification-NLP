import re
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding

def text_cleaning(text):
    """This function removes texts with anomalies such as @NAME, Numbers, brackets and also to convert text into lower case.

    Args:
        text (_type_): Raw text

    Returns:
        _type_: Cleaned text
    """

    # remove tags
    text = re.sub('<.*?>','',text)
    

    # $number and lower case
    text = re.sub('[^a-zA-Z]', ' ', text).lower()

    return text


def lstm_model_creation(num_words,nb_classes,embedding_layer=64,dropout=0.3,num_nodes=64):
    """This function creates LSTM model with embedding layer, 2 LSTM layers and 1 output layer

    Args:
        num_words (_type_): number of vocabulary
        nb_classes (_type_): number of classes
        embedding_layer (int, optional): The number of output of embedding
        dropout (float, optional): The rate of dropout. Defaults to 0.3.
        num_nodes (int, optional): Number of brain nodes/cells. Defaults to 64.

    Returns:
        _type_: Return the Mocel created using Sequential API
    """

    model = Sequential()
    model.add(Embedding(num_words,embedding_layer))
    model.add(Dropout(dropout))
    model.add(LSTM(num_nodes))
    model.add(Dropout(dropout))
    model.add(Dense(nb_classes, activation = 'softmax'))
    model.summary()

    return model