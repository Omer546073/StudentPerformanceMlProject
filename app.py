# app.py

from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the fitted scaler (must be saved from training using the SAME
# StandardScaler instance that transformed X_train)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        features = [float(x) for x in request.form.values()]

        # Convert to numpy array and reshape for prediction
        final_features = np.array(features).reshape(1, -1)

        # Scale the features exactly as done during training
        final_features_scaled = scaler.transform(final_features)

        # Predict
        prediction = model.predict(final_features_scaled)

        return render_template(
            'index.html',
            prediction_text=f'Predicted Value: {prediction[0]:.2f}'
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f'Error: {str(e)}'
        )

if __name__ == "__main__":
    app.run(debug=True)
