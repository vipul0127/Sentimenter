#USED TO CREATE MODEL.PKL AND CVTRANSFORM.PKL USED FOR ANALYSIS OF OTHER FRESH DATA


import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
data = pd.read_csv('/Users/vipul/PycharmProjects/MACHINE_LEARNING/.venv/NLP/a2_RestaurantReviews_FreshDump.tsv', delimiter='\t')


from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
stemmer= PorterStemmer()
stopw= set(stopwords.words('english'))
stopw.remove('not')


import re
corpus=[]
for i in range(0,100):
  review = re.sub('[^a-zA-Z]',' ',data['Review'][i])
  review = review.lower()
  review = review.split()
  review = [stemmer.stem(word) for word in review if not word in stopw]
  review = ' '.join(review)
  corpus.append(review)
print(corpus)


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1420)
x = cv.fit_transform(corpus).toarray()
y = data.iloc[:,-1].values


# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
# import pickle
# pickle.dump(cv,open('cvtransform.pkl','wb'))
# from sklearn.naive_bayes import GaussianNB
# classifier = GaussianNB()
# classifier.fit(x_train,y_train)
# y_train
import joblib
# joblib.dump(classifier,'model.pkl')


# from sklearn.metrics import confusion_matrix,accuracy_score
# y_pred = classifier.predict(x_test)
# cm = confusion_matrix(y_test,y_pred)
# acc= accuracy_score(y_test,y_pred)
# print(cm)
# print(acc)

import pickle
import pandas as pd


with open('cvtransform.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
X_new = vectorizer.transform(corpus).toarray()
print(X_new)


classifier = joblib.load('model.pkl')
yp=classifier.predict(X_new)
data['predicted_label']=yp.tolist()
data.to_csv('output.csv',sep='\t',encoding='UTF-8',index=False)