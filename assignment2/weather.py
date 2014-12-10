
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

X1 = [{'city':'Gothenburg', 'month':'July'},
      {'city':'Gothenburg', 'month':'December'},
      {'city':'Paris', 'month':'July'},
      {'city':'Paris', 'month':'December'}]
Y1 = ['rain', 'rain', 'sun', 'rain']

X2 = [{'city':'Sydney', 'month':'July'},
      {'city':'Sydney', 'month':'December'},
      {'city':'Paris', 'month':'July'},
      {'city':'Paris', 'month':'December'}]
Y2 = ['rain', 'sun', 'sun', 'rain']

X1e = DictVectorizer().fit_transform(X1)
# classifier1 = Perceptron()
classifier1 = LinearSVC()
classifier1.fit(X1e, Y1)
guesses1 = classifier1.predict(X1e)
print(accuracy_score(Y1, guesses1))

X2e = DictVectorizer().fit_transform(X2)
# classifier2 = Perceptron()
classifier2 = LinearSVC()
classifier2.fit(X2e, Y2)
guesses2 = classifier2.predict(X2e)
print(accuracy_score(Y2, guesses2))


