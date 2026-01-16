import pickle
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# Load the breast cancer dataset
dataset = load_breast_cancer()
features = dataset.data
labels = dataset.target

# Split into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=123
)

# Normalize features using MinMaxScaler
normalizer = MinMaxScaler()
X_train_normalized = normalizer.fit_transform(X_train)
X_test_normalized = normalizer.transform(X_test)

# Train Random Forest classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=123, max_depth=10)
classifier.fit(X_train_normalized, y_train)

# Evaluate model accuracy
y_pred = classifier.predict(X_test_normalized)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Set: {test_accuracy:.2%}")

# Save the trained model and normalizer
model_package = {
    'classifier': classifier,
    'normalizer': normalizer,
    'feature_names': dataset.feature_names.tolist()
}

with open('cancer_model.pkl', 'wb') as file:
    pickle.dump(model_package, file)

print("✓ Model saved successfully as 'cancer_model.pkl'")