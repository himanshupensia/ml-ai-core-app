import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# 1. Load Data
# We use return_X_y=True to get the data (X) and labels (y) directly
X, y = load_iris(return_X_y=True)

# 2. Train Model
clf = RandomForestClassifier()
clf.fit(X, y)

# 3. Save Model locally
with open("model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("Success: model.pkl has been created.")