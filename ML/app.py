import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the data
df = pd.read_csv('Book2.csv')

# Create a one-hot encoder
ohe = OneHotEncoder()

# Encode the string features
df_encoded = ohe.fit_transform(df[['Character', 'Diet', 'Exercise Routine', 'Medical History']])

# Convert the sparse matrix to a dense NumPy array
X = df_encoded.toarray()

# Target variable
y = df['Health Issue (Target)'].values

# Create a Random Forest classifier
clf = RandomForestClassifier()

# Fit the classifier to the training data
clf.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the user input from the form
        character = request.form['character']
        diet = request.form['diet']
        exercise = request.form['exercise']
        medical_history = request.form['medical_history']

        # Create a new DataFrame with user input
        new_data = pd.DataFrame({
            'Character': [character],
            'Diet': [diet],
            'Exercise Routine': [exercise],
            'Medical History': [medical_history]
        })

        # Encode the new data using the same encoder
        new_data_encoded = ohe.transform(new_data[['Character', 'Diet', 'Exercise Routine', 'Medical History']])
        X_new = new_data_encoded.toarray()

        # Make a prediction
        predicted_health_issue = clf.predict(X_new)[0]

        return render_template('result.html', character=character, predicted_health_issue=predicted_health_issue)

if __name__ == '__main__':
    app.run(debug=True)
