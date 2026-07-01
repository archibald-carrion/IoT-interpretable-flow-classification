from data_processing import load_preprocess_data
from model_training import train_evaluate
from shap_analysis import shap_analysis

if __name__ == "__main__":
    # preprocess the data
    X_encoded, y_encoded, feature_names, label_encoder, categorical_cols, numeric_cols, ct = load_preprocess_data()

    # train the model and save every artifact needed for later inference
    results, X_train_scaled, X_test_scaled, y_train = train_evaluate(
        X_encoded, y_encoded, feature_names, label_encoder,
        column_transformer=ct, categorical_cols=categorical_cols, numeric_cols=numeric_cols,
        save_model=True, output_dir='artifacts'
    )

    # perform SHAP analysis
    shap_analysis(results, X_train_scaled, y_train, feature_names, label_encoder)