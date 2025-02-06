# SMS Spam Classification

## Discussion

This project involves processing unstructured text data into numerical vectors and creating a classifier to detect whether an SMS message is spam or not. This is implemented making use of Natural Language Processing libraries in python. This model is developed using Google Colab, allowing for direct import of all necessary libraries and dependencies without manual installation. The `.py` file is attached to this repository.

**Note:** If running the notebook in Google Colab, it is advised to upload the dataset to the Files section before executing the code.

### Required Libraries and Dependencies

  - **nltk** – It is the Natural Language Processing Toolkit library which provides various tools and resources to perform natural language processing tasks.
  - **re** – It is the module used for regular expression operations which in this case is used to perform text preprocessing.
  - **PorterStemmer from nltk.stem.porter** – The PorterStemmer is used to perform Stemmming which is text preprocessing technique of reducing the words to their base form.
  - **TfidfVectorizer from sklearn.feature_extraction.text** – It imports the Term Frequency Inverse Document Frequency (TF-IDF) vectorizer which converts text data into numerical features.
  - **train_test_split from sklearn.model_selection** – It is used to split the dataset into training and testing subsets.
  - **KNeighborsClassifier from sklearn.neighbors** – It imports the K-Nearest Neighbor Classifier from scikit-learn to perform classification.
  - **tensorflow** – It imports tensorflow which is used to build and train deep learning models.
  - **keras from tensorflow** – It imports keras library from tensorflow.
  - **Input, Dense from tensorflow.keras.layers** – To import layers from keras.
  - **pandas** – This library is used to import the datasets as a data frame.
  - **numpy** – This library will provide support for arrays, and we can perform mathematical functions on these arrays.
  - **LabelEncoder from sklearn.preprocessing** – This is used to normalize the features in the dataset so that the contain only values between 0 to one less that the number of classes (0 – n_classes-1).
  - **confusion_matrix, accuracy_score, precision_score, recall_score, f1_score** - All of the above mentioned libraries are imported from sklearn.metrics and they are used to analyze the performance of the implementation of the clustering algorithm.

## Dataset

The dataset can be downloaded from, http://archive.ics.uci.edu/dataset/228/sms+spam+collection. The SMSSpamCollection file consists of one message per line with each line containing the first word either as ham which specifies the message to be legitimate or spam which is typically the spam message.

## Methodology

We implement several machine learning algorithms such as K-Means Clustering, KNN, Neural Network, Transformer Block. We could compare the performance metrics obtained in each case to determine a better suited model.
