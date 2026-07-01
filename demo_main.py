import numpy as np
import joblib
from data_processing import load_preprocess_data
from sklearn.model_selection import train_test_split

# Reload data + regenerate the same test split used during training (same random_state)
X_encoded, y_encoded, feature_names, label_encoder, categorical_cols, numeric_cols, ct = load_preprocess_data()
_, X_test, _, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Load your saved artifacts
rf = joblib.load('artifacts/random_forest_model.joblib')
scaler = joblib.load('artifacts/scaler.joblib')

# Pick one "simulated" flow per category — real held-out flows, standing in for live capture
targets = ['MQTT_Publish', 'DOS_SYN_Hping']  # swap in whichever 2 classes you want to show

for class_name in targets:
    class_id = label_encoder.transform([class_name])[0]
    idx = np.where(y_test == class_id)[0][0]

    flow = X_test[idx:idx+1]
    flow_scaled = scaler.transform(flow)

    pred = rf.predict(flow_scaled)[0]
    proba = rf.predict_proba(flow_scaled)[0]
    predicted_label = label_encoder.inverse_transform([pred])[0]

    print(f"Simulated flow (true class: {class_name})")
    print(f"  -> Model prediction: {predicted_label}")
    print(f"  -> Confidence: {proba.max():.2%}\n")