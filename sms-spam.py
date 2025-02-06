"""SMS_Spam_Classification

Done by :
Venkata Janaki Vamsiram Gandham

Initially, we import the necessary libraries

We can observe that each sentence in the file contains ham or spam as its first word which is tab separated by the actual message. So using pandas we can import the dataset by specifying 2 columns namely label and message.
"""

import pandas as pd                         # importing pandas for reading the data as a dataframe


messages = pd.read_csv("SMSSpamCollection", sep='\t', names=["label", "message"])         # importing the data and assigning labels
print(messages.head(5))                     # viewing the first 5 sentences in the data

"""Data cleaning and Data Preprocessing

Removing punctuation marks, stop words.
Also applying Stemming and Lemmatization
"""

import re                                   # importing the regular expressions library
import nltk                                 # importing the nltk library
nltk.download('stopwords')                  # downloading the stopwords from nltk
from nltk.corpus import stopwords           # importing the stopwords from nltk
from nltk.stem.porter import PorterStemmer  # importing PorterStemmer to perform Stemming

ps = PorterStemmer()                        # creating an object to perform Stemming

corpus = []                                 # creating an empty list named as corpus

for i in range(0, len(messages)):
  text = re.sub('[^a-zA-Z]', ' ', messages['message'][i])           # removes all the characters except a-z and A-Z and replaces with a space for every sentence
  text = text.lower()       # removing dupliate words
  text = text.split()       # obtaining a list of words by splitting the sentences

  # now apply stemming to the words in text which are not stopwords
  text = [ps.stem(word) for word in text if not word in stopwords.words('english')]
  text = ' '.join(text)
  corpus.append(text)

"""We can view the corpus here which is basically the text messages after preprocessing."""

print(corpus)

"""Creating a TF-IDF vectorizer to transform the text data into a term-document matrix"""

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

M = tfidf_vectorizer.fit_transform(corpus)

"""Applying the K-Means Clustering algorithm which is created in Project 2.

Initially, we need to import the necessary libraries for our KMeansClustering function.
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

class KMeansClustering():
  def __init__(self, X, num_clusters):                # initilaizing the main properties of the class.
    self.K = num_clusters                             # K is the number of desired clusters.
    self.max_iterations = 1000                        # the maximum number of iterations the algorithm performs.
    self.num_examples, self.num_features = X.shape    # number of data points and features which are derived from the shape of the input data.
    self.plot_figure = True                           # assigning a boolean flag whether a scatter plot is generated to visualize the final clusters.

  def initialize_random_centroids(self, X):           # defining a method to initialize the centroids randomly from the data points in X.
      centroids = np.zeros((self.K, self.num_features)) # initially creating a zero matrix for the centroids.

      for k in range(self.K):
        centroid = X[np.random.choice(range(self.num_examples))]
        centroids[k] = centroid                       # populating the centroids matrix by randomly selecting data points from X.
      return centroids

  def create_clusters(self, X, centroids):
      clusters = [[] for _ in range(self.K)]

      for point_idx, point in enumerate(X):           # computing euclidean distance from the data points to each centroid.
        closest_centroid = np.argmin(np.sqrt(np.sum((point - centroids)**2, axis=1)))
        clusters[closest_centroid].append(point_idx)  # assigning the data point to the cluster of the closest centroid.

      return clusters


  def calcluate_new_centroids(self, clusters, X):     # defining a method to re-calculate the centroids.
      centroids = np.zeros((self.K, self.num_features))

      for idx, cluster in enumerate(clusters):
        new_centroid = np.mean(X[cluster], axis =0)   # new centroid is calculated by taking the mean of all the points in the cluster.
        centroids[idx] = new_centroid

      return centroids

  def predict_cluster(self, clusters, X):             # defining a method to return a label for each data point.
      y_pred = np.zeros(self.num_examples)

      for cluster_idx, cluster in enumerate(clusters):
          for sample_idx in cluster:
              y_pred[sample_idx] = cluster_idx        # indicating each datapoint to which cluster it belongs to.
      return y_pred

  def fit(self, X):                                   # defining the fit method to implement K-means clustering.
      centroids = self.initialize_random_centroids(X) # calling the method to initialize random centroids.

      for it in range(self.max_iterations):
          clusters = self.create_clusters(X, centroids) # calling the method to create clusters.

          previous_centroids = centroids                # storing the intial centroids as previous centroids.
          centroids = self.calcluate_new_centroids(clusters, X) # calculating new centroids.

          diff = centroids - previous_centroids         # calculating the difference between the previous centroids and new centroids.

          if not diff.any():                            # defining a condition to check if there is any difference between previous and new centroids.
              break                                     # if there is no difference then the for loop will break.
      y_pred = self.predict_cluster(clusters, X)        # it return the clusters to which the data points belong to.

      return y_pred

  def performance_metrics(self, y_true, y_pred):        # defining a method to return the performance metrics to evaluate the clustering.
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)
    accuracy = accuracy_score(y_true, y_pred)
    print('\nAccuracy: {0:0.2f}'.format(accuracy*100) + " %")
    precision = precision_score(y_true, y_pred)
    print('\nPrecision: {0:0.2f}'.format(precision, average='macro'))
    recall= recall_score(y_true, y_pred, average='macro')
    print('\nRecall: {0:0.2f}'.format(recall))
    f1score = f1_score(y_true, y_pred, average='macro')
    print('\nF1-score: {0:0.2f}'.format(f1score))

    # Extracting each value from the confusion matrix
    true_positives, false_negatives, false_positives, true_negatives = cm.ravel()
    # Calculating Spam Caught (SC) and Blocked Ham (BH) messages
    sc = (true_positives / (true_positives + false_negatives)) * 100
    print(f"Spam Caught percentage (SC): {sc:.2f}%")

if __name__ == '__main__':
  np.random.seed(42)
  num_clusters = 2
  M = pd.DataFrame(M.toarray(), columns=tfidf_vectorizer.get_feature_names_out())        # converting TF-IDF matrix into a dataframe
  X = M.values                      # loading the term-document matrix values as input X
  Kmeans = KMeansClustering(X, num_clusters)
  y_pred = Kmeans.fit(X)
  label_encoder = LabelEncoder()
  y_true = label_encoder.fit_transform(messages['label'])
  Kmeans.performance_metrics(y_true, y_pred)

"""Now as per the given research paper, we can calculate the Spam Caught (SC) and Blocked Ham (BH) percentages with the help of the confusion matrix.

Applying k-nearest neighbors (KNN) algorithm to classify the spam messages using the term-document matrix M
"""

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

X = M                                             # the TF-IDF matrix
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(messages['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn_classifier = KNeighborsClassifier(n_neighbors=2)
knn_classifier.fit(X_train, y_train)

y_pred = knn_classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
report = classification_report(y_test, y_pred, target_names = ['ham', 'spam'])
cm = confusion_matrix(y_test, y_pred)

print('\nAccuracy: {0:0.2f}'.format(accuracy*100) + " %")
print('\nPrecision: {0:0.2f}'.format(precision))
print('\nRecall: {0:0.2f}'.format(recall))
print('\nF1-score: {0:0.2f}'.format(f1))
print('\nClassification Report')
print(report)
print('\nConfusion Matrix')
print(cm)

# Extracting each value from the confusion matrix

true_positives, false_negatives, false_positives, true_negatives = cm.ravel()

# Calculating Spam Caught (SC) and Blocked Ham (BH) messages

sc = (true_positives / (true_positives + false_negatives)) * 100


print(f"Spam Caught percentage (SC): {sc:.2f}%")

"""Performing Text Classification with Neural Network

Importing necessary libraries
"""

import re                                   # importing the regular expressions library
import nltk                                 # importing the nltk library
nltk.download('stopwords')                  # downloading the stopwords from nltk
from nltk.corpus import stopwords           # importing the stopwords from nltk
from nltk.stem.porter import PorterStemmer  # importing PorterStemmer to perform Stemming
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense

messages = pd.read_csv("SMSSpamCollection", sep='\t', names=["label", "message"])      # import the dataset

# Encode the output labels such as (spam: 1, ham: 0)
le = LabelEncoder()
messages['label'] =  le.fit_transform(messages['label'])

ps = PorterStemmer()                        # creating an object to perform Stemming

corpus = []                                 # creating an empty list named as corpus

for i in range(0, len(messages)):
  text = re.sub('[^a-zA-Z]', ' ', messages['message'][i])           # removes all the characters except a-z and A-Z and replaces with a space for every sentence
  text = text.lower()       # removing dupliate words
  text = text.split()       # obtaining a list of words by splitting the sentences

  # now apply stemming to the words in text which are not stopwords
  text = [ps.stem(word) for word in text if not word in stopwords.words('english')]
  text = ' '.join(text)
  corpus.append(text)

# Creating a TF-IDF vectorizer to transform the text data into a term-document matrix
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

M = tfidf_vectorizer.fit_transform(corpus)

M = pd.DataFrame(M.toarray(), columns=tfidf_vectorizer.get_feature_names_out())        # converting TF-IDF matrix into a dataframe
X = M.values                      # loading the term-document matrix values as input X
y = messages['label'].values      # loading the label values as output

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   # dividing the data into training and testing sets

# Define a custom pooling layer for the term-document matrix
class CustomGlobalAveragePooling1D(keras.layers.Layer):
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=1)

# Using tensorflow and keras, we define a neural network model.

input_layer = Input(shape=(X_train.shape[1],))
hidden_layer = Dense(32, activation='relu')(input_layer)
dropout_layer = keras.layers.Dropout(0.3)(hidden_layer)
output_layer = Dense(1, activation='sigmoid')(dropout_layer)

model = keras.Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# training the model using the fit method.

model.fit(X_train, y_train, validation_split=0.2, epochs=10)

# evaluating the trained model to make predictions on the test set and roudning off the predictions to the nearest integer which in this case is (0 or 1)

y_pred = model.predict(X_test)
y_pred = np.round(y_pred)

# Evaluating the performance of the model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
report = classification_report(y_test, y_pred, target_names = ['ham', 'spam'])
cm = confusion_matrix(y_test, y_pred)

print('\nAccuracy: {0:0.2f}'.format(accuracy*100) + " %")
print('\nPrecision: {0:0.2f}'.format(precision))
print('\nRecall: {0:0.2f}'.format(recall))
print('\nF1-score: {0:0.2f}'.format(f1))
print('\nClassification Report')
print(report)
print('\nConfusion Matrix')
print(cm)
# Extracting each value from the confusion matrix

true_positives, false_negatives, false_positives, true_negatives = cm.ravel()

# Calculating Spam Caught (SC) and Blocked Ham (BH) messages

sc = (true_positives / (true_positives + false_negatives)) * 100

print(f"Spam Caught percentage (SC): {sc:.2f}%")

"""Performing text classification using Transformer Block"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers

# Load the SMSSpamCollection dataset
messages = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])

# Using LabeEncoder we will encode the labels such as(spam: 1, ham: 0)
label_encoder = LabelEncoder()
messages['label'] = label_encoder.fit_transform(messages['label'])

# Using train_test_split we can split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(messages['message'], messages['label'], test_size=0.2, random_state=42)

# Using Tokenizer to perform text preprocessing
tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(X_train)

# Converting the text data to sequences
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# Pading the sequences to ensure consistent length
max_len = max(len(seq) for seq in X_train)
X_train = keras.preprocessing.sequence.pad_sequences(X_train, padding='post')
X_test = keras.preprocessing.sequence.pad_sequences(X_test, padding='post')

# Defining the function to build the Transformer model
def transformer_model(max_len, vocab_size):
    inputs = layers.Input(shape=(max_len,))                                                   # defining input layer
    embedding_layer = layers.Embedding(input_dim=vocab_size, output_dim=128)(inputs)          # defining embedding layer
    transformer_block = TransformerBlock(128, 8, 64)(embedding_layer)                         # defining transformer block
    t = layers.GlobalAveragePooling1D()(transformer_block)
    t = layers.Dropout(0.3)(t)                                                                # dropout layer to prevent overfitting
    t = layers.Dense(20, activation="relu")(t)
    t = layers.Dropout(0.3)(t)
    outputs = layers.Dense(1, activation="sigmoid")(t)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Defining the Transformer block class
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Training and Testing the model
max_len = X_train.shape[1]
vocab_size = len(tokenizer.word_index) + 1
model = transformer_model(max_len, vocab_size)
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

accuracy = transformer_model.evaluate(X_test, y_test)[1]
print('\nAccuracy: {0:0.2f}'.format(accuracy*100) + " %")
