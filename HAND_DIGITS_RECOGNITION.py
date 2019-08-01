import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
digits = load_digits

#It is a type of dictionary andit is 2-dimensional
digits
dir(digits)

#Showing images of digits
plt.gray()
for i in range(9):
    plt.matshow(digits.images[0])

#Showing target array
digits.target[:5]

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

#Performing testing and training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target,test_size=0.2)

#Total train samples
len(X_train)
#Total test samples
len(X_test)

#Model ready for training on training samples
model.fit(X_train,y_train)

#Now testing on test samples

#Model Prediction
model.predict(X_test)

#Real output
y_test

model.score(X_test, y_test)

y_predicted = model.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_predicted)
cm

