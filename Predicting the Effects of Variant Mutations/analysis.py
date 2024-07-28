import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

with open('/Users/jborri/Documents/GitHub/Mitochondrial-Haplogroup-Mutations/Predicting the Effects of Variant Mutations/P01308.json', 'r') as file:
    data = json.load(file)

def preprocess_data(data):
    variants = data['features']
    records = []
    for variant in variants:
        if variant['type'] == 'VARIANT':
            record = {
                'alt_seq': variant['alternativeSequence'],
                'position': variant['begin'],
                'genomic_location': variant.get('genomicLocation', [None])[0],
                'consequence_type': variant['consequenceType'],
                'wild_type': variant['wildType'],
                'mutated_type': variant['mutatedType'],
                'polyphen': next((pred['predictionValType'] for pred in variant.get('predictions', []) if pred['predAlgorithmNameType'] == 'PolyPhen'), 'unknown'),
                'sift': next((pred['predictionValType'] for pred in variant.get('predictions', []) if pred['predAlgorithmNameType'] == 'SIFT'), 'unknown'),
                'pathogenicity': next((sign['type'] for sign in variant.get('clinicalSignificances', [])), 'unknown')
            }
            records.append(record)
    return pd.DataFrame(records)

data = json.load(file)
df = preprocess_data(data)

df_encoded = pd.get_dummies(df, columns=['consequence_type', 'wild_type', 'mutated_type', 'polyphen', 'sift'])

X = df_encoded.drop(columns=['pathogenicity'])
y = df_encoded['pathogenicity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)