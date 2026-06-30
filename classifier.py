from data_processing import load_preprocess_data
from model_training import train_evaluate
from shap_analysis import shap_analysis

if __name__ == "__main__":
    # preprocess the data
    X_encoded, y_encoded, feature_names, label_encoder, categorical_cols, numeric_cols = load_preprocess_data()
    # train the model
    results, X_train_scaled, X_test_scaled = train_evaluate(
        X_encoded, y_encoded, feature_names, label_encoder
    )
    #perform SHAP analysis
    shap_analysis(results, X_train_scaled, feature_names, label_encoder)
