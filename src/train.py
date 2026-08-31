from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data # shape (150,4)
y = iris.target # shape (150,)

print(iris.feature_names, iris.target_names)

from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier (random_state=42)

model.fit(X_train, y_train)
DecisionTreeClassifier(random_state=42)

y_pred = model.predict(X_test)

print("Predictions:", y_pred[:5])
print("True labels:", y_test[:5])


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("accuracy:", accuracy)
accuracy: 1.0

from sklearn.neighbors import KNeighborsClassifier

model2 = KNeighborsClassifier(n_neighbors=5)

model2.fit(X_train, y_train)
KNeighborsClassifier()

y_pred2 = model2.predict(X_test)

print("k-NN accuracy:", accuracy_score(y_test, y_pred2))

model = DecisionTreeClassifier(max_depth=3, random_state=42)

model.fit(X_train, y_train)






import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay( confusion_matrix=cm, display_labels=iris.target_names)

disp.plot()

plt.title("Decision Tree Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

# Save the trained model
joblib.dump(model, "outputs/model.joblib")