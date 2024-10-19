import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class KeystrokeDynamicsModel:
    def __init__(self):
        self.model = RandomForestClassifier()
    
    def preprocess_data(self, keystroke_data):
        # Extract relevant features from keystroke data
        # Example: Calculate key hold times and interval times
        features = []
        for data in keystroke_data:
            hold_times = [t['timestamp'] - data['timestamps'][0] for t in data['timestamps']]
            interval_times = [data['timestamps'][i+1] - data['timestamps'][i] for i in range(len(data['timestamps'])-1)]
            features.append(hold_times + interval_times)
        return np.array(features)

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained. Accuracy: {accuracy}")

    def predict(self, X):
        return self.model.predict(X)

keystroke_model = KeystrokeDynamicsModel()