#Sentiment analysis stanford

#The data is a CSV with emoticons removed. Data file format has 6 fields:
#0 - the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
#1 - the id of the tweet (2087)
#2 - the date of the tweet (Sat May 16 23:58:44 UTC 2009)
#3 - the query (lyx). If there is no query, then this value is NO_QUERY.
#4 - the user that tweeted (robotickilldozr)
#5 - the text of the tweet (Lyx is cool)

################################################
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import pandas_profiling
from matplotlib import pyplot

import string

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import re
import nltk

from nltk.corpus import stopwords

import string

from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split

import time

#Load train dataset
train_data = pd.read_csv("C:\\Users\\kode surendra aba\\Desktop\\Data science\\python\\Sample projects\\sentiment_analysis_tweet_nlp\\training.1600000.processed.noemoticon.csv",header=None,encoding = "ISO-8859-1")

train_data.shape #(1600000, 6)

train_data.head()

train_data = train_data.rename(columns={0: 'polarity', 1: 'id', 2: 'date', 3: 'query_type', 4: 'user',5: 'text'})

#Loading test data
test_data = pd.read_csv("C:\\Users\\kode surendra aba\\Desktop\\Data science\\python\\Sample projects\\sentiment_analysis_tweet_nlp\\testdata.manual.2009.06.14.csv",header=None,encoding = "ISO-8859-1")

test_data.shape #(498, 6)

test_data.head()

test_data = test_data.rename(columns={0: 'polarity', 1: 'id', 2: 'date', 3: 'query_type', 4: 'user',5: 'text'})

#Merging train and test data to clean the dataset
data = train_data.append(pd.DataFrame(data = test_data), ignore_index=True)

#Consider only 1% of dataset due to memory constraints
data = data.sample(frac = 0.01) 

data.shape #(1600498, 6)

list(data) #['polarity', 'id', 'date', 'query_type', 'user', 'text']

#Removing duplicates
data.drop_duplicates(inplace = True)

data.shape #No duplicates found in data

print (pd.DataFrame(data.isnull().sum())) #No null values found in data

#Column id
data.id.value_counts() #Column id has maximum 2frequencies of any id so there are lot of unique values here which is useless in algorithms

del data['id']

#Column date
data.date.value_counts()

#separate elements date column
data['date'] = data['date'].map(lambda date:re.sub('\W+', ' ',date)).apply(lambda x: (x.lower()).split())
#The division will be as follows ['mon', 'apr', '06', '22', '19', '45', 'pdt', '2009']

#extracting weekday from date
data.loc[:, 'weekday'] = data.date.map(lambda x: x[0])
data.weekday.value_counts()


#extracting month from date
data.loc[:, 'month'] = data.date.map(lambda x: x[1])
data.month.value_counts()

#extracting day from date
data.loc[:, 'day'] = data.date.map(lambda x: x[2])
data['day'] = pd.to_numeric(data['day']) #Convert day column to numeric values
data.day.value_counts()

#convert days to bins of different monthframes of a month like month_start, month_mid and month_end
conditions = [
    (data['day'] >=0) & (data['day'] <= 10),
    (data['day'] >=11) & (data['day'] <= 20)
    ]
choices = ['month_start','month_mid']
data['monthframe'] = np.select(conditions, choices, default='month_end')

data.monthframe.value_counts()

#Remove day column as we don't need it now after binning it
del data['day']


#extracting hour from date
data.loc[:, 'hour'] = data.date.map(lambda x: x[3])
data['hour'] = pd.to_numeric(data['hour']) #Convert hour column to numeric values
data.hour.value_counts()

#convert hours to bins of different timeframes of a day like marning, evening , afternoon and night
conditions = [
    (data['hour'] >=0) & (data['hour'] <= 5),
    (data['hour'] >=6) & (data['hour'] <= 12),
    (data['hour'] >=13) & (data['hour'] <= 16),
    (data['hour'] >=17) & (data['hour'] <= 20)
    ]
choices = ['night', 'morning', 'afternoon','evening']
data['timeframe'] = np.select(conditions, choices, default='night')

data.timeframe.value_counts()

#Remove hour column as we don't need it now after binning it
del data['hour']


#extracting year from date
data.loc[:, 'year'] = data.date.map(lambda x: x[7])
data.year.value_counts()
#The data contains just one year so we remove year column
del data['year']

#We remove date column from data since it is of no use now
del data['date']

#Column user
data.user.value_counts() #660120 unique users which are also much unique so we remove user also
del data['user']

#Column query
data.query_type.value_counts() #1600000 are NO_QUERY so we remove this column too
del data['query_type']



##### Convert to dummy variables

#Create weekday column to numeric dummy variables
df_weekday = pd.get_dummies(data['weekday'])

#Create month column to numeric dummy variables
df_month = pd.get_dummies(data['month'])

#Create monthframe column to numeric dummy variables
df_monthframe = pd.get_dummies(data['monthframe'])

#Create timeframe column to numeric dummy variables
df_timeframe = pd.get_dummies(data['timeframe'])

#merging dummy varibles to form a dataframe
data = pd.concat([data['polarity'].reset_index(drop=True),df_weekday.reset_index(drop=True),df_month.reset_index(drop=True),df_monthframe.reset_index(drop=True),df_timeframe.reset_index(drop=True),data['text'].reset_index(drop=True)],axis=1)
list(data)

#Working with text column
#NLP coming ahead

#Column text
#separate elements of text column in lists, also removing hashtags. mentionids and urls if any
data['p_text'] = data['text'].map(lambda text:re.sub('(@[A-Za-z0-9]+)|(\w+:\/\/\S+)|(#[A-Za-z0-9]+)|([^A-Za-z\'\"]+)', ' ',text)).apply(lambda x: (x.lower()).split())


        
#joining p_text aain for removing stopwords and punctuation later
data['p_text'] = data['p_text'].apply(lambda x: " ".join([word for word in x]))


stopwords = nltk.corpus.stopwords.words('english')

#wn = nltk.WordNetLemmatizer()
ps = nltk.PorterStemmer()

#Removing stopwords and punctuations
def clean_text(text):
    text = "".join([word for word in text if word not in string.punctuation])
    tokens = re.split(' ', text)
    text = [word for word in tokens if word not in stopwords]
    text = [ps.stem(word) for word in text]
    #text = [wn.lemmatize(word) for word in text]
    return text


#Removing original text column
del data['text']

list(data)
data.shape


#Vectorizing processed text column i.e. p_text
tfidf_vect = TfidfVectorizer(analyzer=clean_text)
X_tfidf = tfidf_vect.fit_transform(data['p_text'])
print(X_tfidf.shape)

print(tfidf_vect.get_feature_names())

X_tfidf_df = pd.DataFrame(X_tfidf.toarray())
X_tfidf_df.columns = tfidf_vect.get_feature_names()

#Taking independent variables together
X_features = pd.concat([data[data.columns[1:18]].reset_index(drop=True),X_tfidf_df.reset_index(drop=True)], axis=1)

X_features.head()

#Divide data in train and test
X_train, X_test, y_train, y_test = train_test_split(X_features, data['polarity'], test_size=0.2)

rf = RandomForestClassifier(n_estimators=50, max_depth=20, n_jobs=-1)
rf_model = rf.fit(X_train, y_train)

#10 Most important factors affecting our model
sorted(zip(rf_model.feature_importances_, X_train.columns), reverse=True)[0:10]

y_pred = rf_model.predict(X_test)
precision, recall, fscore, support = score(y_test, y_pred, pos_label=0, average='binary')

print('Precision: {} / Recall: {} / Accuracy: {}'.format(round(precision, 3),
                                                        round(recall, 3),
                                                        round((y_pred==y_test).sum() / len(y_pred),3)))

#Checing best hyperparameter value to choose for better results
def train_RF(n_est, depth):
    rf = RandomForestClassifier(n_estimators=n_est, max_depth=depth, n_jobs=-1)
    rf_model = rf.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    precision, recall, fscore, support = score(y_test, y_pred, average='micro')
    print('Est: {} / Depth: {} ---- Precision: {} / Recall: {} / Accuracy: {}'.format(
        n_est, depth, round(precision, 3), round(recall, 3),
        round((y_pred==y_test).sum() / len(y_pred), 3)))
    
for n_est in [50, 100, 300, 400]:
    for depth in [10, 20, 30, None]:
        train_RF(n_est, depth)

#Est: 100 / Depth: None ---- Precision: 0.767 / Recall: 0.767 / Accuracy: 0.767
 
#Choosing best hyperparameters       
rf = RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1)

start = time.time()
rf_model = rf.fit(X_train, y_train)
end = time.time()
fit_time = (end - start)

start = time.time()
y_pred = rf_model.predict(X_test)
end = time.time()
pred_time = (end - start)

precision, recall, fscore, train_support = score(y_test, y_pred, average='micro')
print('Fit time: {} / Predict time: {} ---- Precision: {} / Recall: {} / Accuracy: {}'.format(
    round(fit_time, 3), round(pred_time, 3), round(precision, 3), round(recall, 3), round((y_pred==y_test).sum()/len(y_pred), 3)))
    
#Fit time: 23.914 / Predict time: 0.594 ---- Precision: 0.768 / Recall: 0.768 / Accuracy: 0.768
        
        


