import json
import pandas as pd
import warnings
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

with open('/Users/jborri/Documents/GitHub/Mitochondrial-Haplogroup-Mutations/Predicting the Effects of Variant Mutations/P01308.json', 'r') as file:
    data = json.load(file)

def preprocess_data(data):
    variants = data['features']
    records = []
    for variant in variants:
        if variant['type'] == 'VARIANT':
            record = {
                'alt_seq': variant.get('alternativeSequence', 'unknown'), # Changed line
                'position': variant['begin'],
                'genomic_location': variant.get('genomicLocation', [None])[0],
                'consequence_type': variant['consequenceType'],
                'wild_type': variant['wildType'],
                'mutated_type': variant.get('mutatedType', 'unknown'),
                'polyphen': next((pred['predictionValType'] for pred in variant.get('predictions', []) if pred['predAlgorithmNameType'] == 'PolyPhen'), 'unknown'),
                'sift': next((pred['predictionValType'] for pred in variant.get('predictions', []) if pred['predAlgorithmNameType'] == 'SIFT'), 'unknown'),
                'pathogenicity': next((sign['type'] for sign in variant.get('clinicalSignificances', [])), 'unknown')
            }
            records.append(record)
    return pd.DataFrame(records)


df = preprocess_data(data)

# Convert all columns to string type to ensure one-hot encoding works correctly
df = df.astype(str)

df_encoded = pd.get_dummies(df, columns=['consequence_type', 'wild_type', 'mutated_type', 'polyphen', 'sift','alt_seq', 'genomic_location']) # Apply one-hot encoding to all non-numeric columns

# Standardize numerical features
scaler = StandardScaler()
df['position'] = scaler.fit_transform(df[['position']])

X = df_encoded.drop(columns=['pathogenicity'])
y = df_encoded['pathogenicity']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

def predict_pathogenicity(variant):
    variant_encoded = pd.get_dummies(pd.DataFrame([variant]), columns=['consequence_type', 'wild_type', 'mutated_type', 'polyphen', 'sift', 'alt_seq', 'genomic_location']) # Include 'alt_seq' and 'genomic_location' in the one-hot encoding
    
    # Get missing columns in the encoded variant
    missing_cols = set(X_train.columns) - set(variant_encoded.columns)
    # Add a missing column in the encoded variant with a value of 0
    for c in missing_cols:
        variant_encoded[c] = 0
    # Ensure the order of column in the encoded variant is in the same order than in X_train
    variant_encoded = variant_encoded[X_train.columns]
    
    return model.predict(variant_encoded)

new_variant = {
    'alt_seq': 'I',
    'position': 1,
    'genomic_location': 'NC_000011.10:g.2160969C>A',
    'consequence_type': 'missense',
    'wild_type': 'M',
    'mutated_type': 'I',
    'polyphen': 'unknown',
    'sift': 'deleterious'
}
print("Predicted Pathogenicity:", predict_pathogenicity(new_variant))


# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred1 = best_model.predict(X_test)
print(y_pred1)

print("Accuracy:", accuracy_score(y_test, y_pred1))
print("Classification Report:(best model)")
print(classification_report(y_test, y_pred1))

def generate_variant(df):
  """Generates a new variant with random values based on existing data."""

  variant = {}

  # Sample alt_seq from existing values
  variant['alt_seq'] = random.choice(df['alt_seq'].unique())

  # Generate random position within observed range
  min_pos = int(df['position'].min()) # Convert min_pos to integer
  max_pos = int(df['position'].max()) # Convert max_pos to integer
  variant['position'] = random.randint(min_pos, max_pos)

  # Sample genomic_location from existing values
  variant['genomic_location'] = random.choice(df['genomic_location'].unique())

  # Sample consequence_type from existing values
  variant['consequence_type'] = random.choice(df['consequence_type'].unique())

  # Sample wild_type from existing values
  variant['wild_type'] = random.choice(df['wild_type'].unique())

  # Sample mutated_type from existing values
  variant['mutated_type'] = random.choice(df['mutated_type'].unique())

  # Sample polyphen from existing values
  variant['polyphen'] = random.choice(df['polyphen'].unique())

  # Sample sift from existing values
  variant['sift'] = random.choice(df['sift'].unique())

  return variant

for _ in range(5):  # Generate 5 variants
    new_variant = generate_variant(df)
    print("Generated Variant:", new_variant)
    prediction = predict_pathogenicity(new_variant)
    print("Predicted Pathogenicity:", prediction)
    print("---")

    from sklearn.utils import resample

n_iterations = 100
scores = []
for i in range(n_iterations):
    X_train, y_train = resample(X, y)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))

print("Bootstrap scores:", scores)

#Comparing best model 

from sklearn.utils import resample

n_iterations = 100
scores = []
for i in range(n_iterations):
    X_train, y_train = resample(X, y)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))

print("Bootstrap scores(best model):", scores)

from sklearn.model_selection import cross_val_score

# Example with 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation scores:", scores)
print("Average accuracy:", scores.mean())

#comparing best model

from sklearn.model_selection import cross_val_score

# Example with 5-fold cross-validation
scores = cross_val_score(best_model, X, y, cv=5)
print("Cross-validation scores(best model):", scores)
print("Average accuracy:", scores.mean())