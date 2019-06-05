import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re
import nltk


# Access stop words
stop_words = nltk.corpus.stopwords.words('english')

# Remove word stems using a Porter stemmer
porter = nltk.PorterStemmer()

#Access the data 
df = pd.read_table('Data/SMSSpamCollection/SMSSpamCollection.txt', header=None)

# Store the target variable
y = df[0]

#Label Encoder
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Store the SMS message data
raw_text = df[1]

#Preprocessing the messy string
def preprocess_text(messy_string):
    assert(type(messy_string) == str)
    cleaned = re.sub(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr', messy_string)
    cleaned = re.sub(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr',
                     cleaned)
    cleaned = re.sub(r'£|\$', 'moneysymb', cleaned)
    cleaned = re.sub(
        r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
        'phonenumbr', cleaned)
    cleaned = re.sub(r'\d+(\.\d+)?', 'numbr', cleaned)
    cleaned = re.sub(r'[^\w\d\s]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'^\s+|\s+?$', '', cleaned.lower())
    return ' '.join(
        porter.stem(term) 
        for term in cleaned.split()
        if term not in set(stop_words)
    )

#Applying preprocess_text function to all the text data we have and storing it in a new column called 'documents'
df['documents'] = df[1].apply(preprocess_text)


x = df['documents']

#Train test split
from sklearn.model_selection import train_test_split
xTrain, xTest, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Import the libraries we need
import pandas as pd

#  Design the Vocabulary
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer()
count_vectorizer.fit(xTrain)

# Create the Bag-of-Words Model
X_train = count_vectorizer.transform(xTrain)
X_test = count_vectorizer.transform(xTest)

# Training a LogistricRegression model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)

print(score)