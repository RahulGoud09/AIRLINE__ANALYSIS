import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pickle

# Load your dataset (replace with actual airline dataset)
# Dataset should have 'text' and 'sentiment' columns (-1, 0, +1)
data = pd.DataFrame({
    'text': [
        'Good flight!',
        'Terrible service',
        'Average experience',
        'Loved the food',
        'Late departure'
    ],
    'sentiment': [1, -1, 0, 1, -1]
})

# Preprocessing
X = data['text']
y = data['sentiment']

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Model training
model = LinearSVC()
model.fit(X_vec, y)

# Save model and vectorizer
with open('model.pickle', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pickle', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model training completed and files saved!")