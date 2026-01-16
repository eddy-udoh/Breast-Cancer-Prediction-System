import pickle
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# 1. Load data
data = load_breast_cancer()
# We use all features for training to keep the model accurate
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# 2. Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 3. Train SVM Model
model = SVC(kernel='linear', random_state=42)
model.fit(X_train_scaled, y_train)

# 4. Save model and scaler as model.h5
with open('model.h5', 'wb') as f:
    pickle.dump({'model': model, 'scaler': scaler}, f)

print("Success! 'model.h5' for Breast Cancer (SVM) has been created.")