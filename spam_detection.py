import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

#load dataset
df=pd.read_csv('spam_detection.csv')
print(df)
print(df.isnull().sum())

# Features
x=df['message']
y=df['label']
print(x)
print(y)

# Convert  text to number
vt=CountVectorizer()
x_vt=vt.fit_transform(x)

# train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_vt,y,test_size=0.25,random_state=42)

# Train model
model=MultinomialNB()
model.fit(x_train,y_train)

# Prediction
y_pred=model.predict(x_test)

# Accuracy
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy score:",accuracy)

# Confusion matrix
print("---Confusion matrix---")
print(confusion_matrix(y_test,y_pred))

# Classification Report
print("---Classification Report---")
print(classification_report(y_test,y_pred))

# User input
message=input("Enter your message:")
message_vactor=vt.transform([message])

result=model.predict(message_vactor)
if result[0]=='ham':
    print("Ham")
else:
    print("Spam")

