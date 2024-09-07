from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained models
model_SSA = joblib.load('best_rf_SSA.pkl')
model_KVP = joblib.load('best_rf_KVP.pkl')
model_SCSS = joblib.load('best_rf_SCSS.pkl')

# Define features
features = ['gender_ratio', 'population', 'avg_age', 'avg_income', 'literacy_rate', 'agriculture_occupation_ratio']

# Define feature importance weights
feature_importance = {
    'SSA': {'gender_ratio': 2, 'avg_age': 0, 'avg_income': 0, 'literacy_rate': 0, 'agriculture_occupation_ratio': 0},
    'KVP': {'gender_ratio': 0, 'avg_age': 0, 'avg_income': 0, 'literacy_rate': 0, 'agriculture_occupation_ratio': 2},
    'SCSS': {'gender_ratio': 0, 'avg_age': 1.0, 'avg_income': 0, 'literacy_rate': 0, 'agriculture_occupation_ratio': 0}
}

# Function to apply feature importance scaling
def apply_feature_importance(X, scheme):
    weights = feature_importance[scheme]
    X_weighted = X.copy()
    for feature, weight in weights.items():
        X_weighted[feature] = X_weighted[feature] * weight
    return X_weighted

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json(force=True)
    
    # Convert new city data into a DataFrame
    new_city_features = pd.DataFrame([data], columns=features)
    
    # Apply feature importance to new city data
    new_city_SSA = apply_feature_importance(new_city_features, 'SSA')
    new_city_KVP = apply_feature_importance(new_city_features, 'KVP')
    new_city_SCSS = apply_feature_importance(new_city_features, 'SCSS')

    # Predict participation for the new city using tuned models
    participation_SSA = model_SSA.predict(new_city_SSA)[0]
    participation_KVP = model_KVP.predict(new_city_KVP)[0]
    participation_SCSS = model_SCSS.predict(new_city_SCSS)[0]

    # Prepare the response
    response = {
        "SSA": participation_SSA,
        "KVP": participation_KVP,
        "SCSS": participation_SCSS
    }

    # Function to calculate success probability based on participation percentage
    def calculate_success_probability(participation_percentage):
        if participation_percentage >= 30:
            return 100  # Maximum probability for 30% or more participation
        elif participation_percentage <= 0:
            return 0  # Minimum probability for 0% participation
        else:
            return (participation_percentage / 30) * 100  # Scale between 0% and 30%

    # Calculate success probabilities
    success_probability = {scheme: calculate_success_probability(p) for scheme, p in response.items()}

    # Convert np.float64 to native Python float
    success_probabilities = {k: float(v) for k, v in success_probability.items()}
    sorted_probabilities = dict(sorted(success_probabilities.items(), key=lambda x: x[1], reverse=True))
    print(sorted_probabilities)
    # Return the success probabilities as JSON
    return jsonify(sorted_probabilities)

if __name__ == '__main__':
    app.run(debug=True)
