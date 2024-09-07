import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from sklearn.model_selection import cross_val_score
import joblib
# Load environment variables
load_dotenv()
MONGO_URI = os.getenv('MONGO_URI')

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client['Model_train']
collection = db['new_participation']

# Load data from MongoDB
data = pd.DataFrame(list(collection.find()))

# Normalize and flatten nested participation data
schemes_participation = pd.json_normalize(data['schemes_participation'])
data = data.drop(columns=['schemes_participation'])
data = pd.concat([data, schemes_participation], axis=1)

# Define features and target variables
features = ['gender_ratio', 'population', 'avg_age', 'avg_income', 'literacy_rate', 'agriculture_occupation_ratio']
targets = ['SSA', 'KVP', 'SCSS']

# Define custom feature importance weights

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

# Apply the feature importance weights to create new feature sets for each scheme
X_SSA = apply_feature_importance(data[features], 'SSA')
X_KVP = apply_feature_importance(data[features], 'KVP')
X_SCSS = apply_feature_importance(data[features], 'SCSS')

# Split data into training and testing sets for each scheme
X_train_SSA, X_test_SSA, y_train_SSA, y_test_SSA = train_test_split(X_SSA, data['SSA'], test_size=0.2, random_state=42)
X_train_KVP, X_test_KVP, y_train_KVP, y_test_KVP = train_test_split(X_KVP, data['KVP'], test_size=0.2, random_state=42)

X_train_SCSS, X_test_SCSS, y_train_SCSS, y_test_SCSS = train_test_split(X_SCSS, data['SCSS'], test_size=0.2, random_state=42)

# Parameter grid for RandomForestRegressor
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
}

# Function to perform hyperparameter tuning and return the best model
def hyperparameter_tuning(X_train, y_train, scheme_name):
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    print(f"Best hyperparameters for {scheme_name}: {grid_search.best_params_}")
    return best_rf

# Function to perform cross-validation
def cross_validate_model(model, X, y, scheme_name):
    # Perform cross-validation
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    rmse_scores = (-scores) ** 0.5  # Convert negative MSE to RMSE
    print(f"{scheme_name} Cross-Validation RMSE Scores: {rmse_scores}")
    print(f"{scheme_name} Mean RMSE: {rmse_scores.mean()}")
    return rmse_scores.mean()

# Tuning for SSA
best_rf_SSA = hyperparameter_tuning(X_train_SSA, y_train_SSA, 'SSA')
cross_validate_model(best_rf_SSA, X_SSA, data['SSA'], 'SSA')

# Tuning for KVP
best_rf_KVP = hyperparameter_tuning(X_train_KVP, y_train_KVP, 'KVP')
cross_validate_model(best_rf_KVP, X_KVP, data['KVP'], 'KVP')



# Tuning for SCSS
best_rf_SCSS = hyperparameter_tuning(X_train_SCSS, y_train_SCSS, 'SCSS')
cross_validate_model(best_rf_SCSS, X_SCSS, data['SCSS'], 'SCSS')

# Evaluate the models after tuning
def evaluate_model(model, X_test, y_test, scheme_name):
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    print(f"{scheme_name} Model Evaluation with Hyperparameter Tuning: RMSE: {rmse}, R2 Score: {r2}")
    return rmse, r2

# Evaluate all models
evaluate_model(best_rf_SSA, X_test_SSA, y_test_SSA, 'SSA')
evaluate_model(best_rf_KVP, X_test_KVP, y_test_KVP, 'KVP')
evaluate_model(best_rf_SCSS, X_test_SCSS, y_test_SCSS, 'SCSS')

# Example: Predict for a new city after tuning
 # 40 to 60
   #   0.3-0.78
new_city = {
    "gender_ratio": 0.3,
    "population": 50000,
    "avg_age": 35,
   
    "avg_income": 75000,
    "literacy_rate": 95,
    "agriculture_occupation_ratio": 0.9  
  
}

# Convert new city data into a DataFrame
new_city_features = pd.DataFrame([new_city], columns=features)

# Apply feature importance to new city data
new_city_SSA = apply_feature_importance(new_city_features, 'SSA')
new_city_KVP = apply_feature_importance(new_city_features, 'KVP')
new_city_SCSS = apply_feature_importance(new_city_features, 'SCSS')



# Predict participation for the new city using tuned models
participation_SSA = best_rf_SSA.predict(new_city_SSA)[0]
participation_KVP = best_rf_KVP.predict(new_city_KVP)[0]

participation_SCSS = best_rf_SCSS.predict(new_city_SCSS)[0]

# Normalize by population
population = new_city["population"]

response = {
    "SSA": participation_SSA ,
    "KVP": participation_KVP ,
   
    "SCSS": participation_SCSS ,
}
print(response)

def calculate_success_probability(participation_percentage):
    if participation_percentage >= 30:
        return 100  # Maximum probability for 15% or more participation
    elif participation_percentage <= 0:
        return 0  # Minimum probability for 0% participation
    else:
        return (participation_percentage / 30) * 100  # Scale between 0% and 15%

success_probability = {}
for scheme, participation in response.items():
    success_probability[scheme] = calculate_success_probability(participation)



# Convert np.float64 to native Python float
success_probabilities = {k: float(v) for k, v in success_probability.items()}

# Sort success probabilities
sorted_probabilities = dict(sorted(success_probabilities.items(), key=lambda x: x[1], reverse=True))

# Print sorted success probabilities
print("Sorted success probabilities for the new city:", sorted_probabilities)

joblib.dump(best_rf_SSA, 'best_rf_SSA.pkl')
joblib.dump(best_rf_KVP, 'best_rf_KVP.pkl')
joblib.dump(best_rf_SCSS, 'best_rf_SCSS.pkl')



