from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

print("Loading dataset...")
mndata = read_csv("/home/ashika29/Downloads/train.csv")
images = mndata.iloc[:,1:]
labels = mndata.iloc[:,0]

clf = KNeighborsClassifier()

# Train on the first 10000 images:
train_x = images[:10000]
train_y = labels[:10000]

print("Train model")
clf.fit(train_x, train_y)

# Test on the next 100 images:
test_x = images[10000:10100]
expected = labels[10000:10100].tolist()

print("Compute predictions")
predicted = clf.predict(test_x)

print("Accuracy: ", accuracy_score(expected, predicted))
