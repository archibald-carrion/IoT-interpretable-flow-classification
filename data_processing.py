from ucimlrepo import fetch_ucirepo
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer


def load_preprocess_data():
    """Fetch, encode, and return dataset features and labels."""
    rt_iot2022 = fetch_ucirepo(id=942)
    X = rt_iot2022.data.features
    y = rt_iot2022.data.targets

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns

    print("--- Original Features ---")
    print("Categorical columns (to be one-hot encoded):")
    print(categorical_cols.tolist())
    print("\nNumeric columns (unchanged):")
    print(numeric_cols.tolist())

    print("\n--- Attack Types in Dataset ---")
    attack_types = np.unique(y['Attack_type'])
    print(attack_types.tolist())

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y['Attack_type'])

    ct = ColumnTransformer(
        [('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],
        remainder='passthrough'
    )
    X_encoded = ct.fit_transform(X)

    encoded_feature_names = (
        ct.named_transformers_['one_hot_encoder'].get_feature_names_out(categorical_cols)
        .tolist() + numeric_cols.tolist()
    )

    print("\n--- One-Hot Encoding Details ---")
    print(f"Original number of categorical columns: {len(categorical_cols)}")
    print(f"Number of numeric columns: {len(numeric_cols)}")
    print(f"Total features after one-hot encoding: {len(encoded_feature_names)}")

    one_hot_feature_names = ct.named_transformers_['one_hot_encoder'].get_feature_names_out(categorical_cols)
    print("\nOne-hot encoded features (replacing original categorical columns):")
    for col in categorical_cols:
        derived_features = [f for f in one_hot_feature_names if f.startswith(col)]
        print(f"- Original column '{col}' was expanded into: {derived_features}")

    return X_encoded, y_encoded, encoded_feature_names, label_encoder, categorical_cols, numeric_cols, ct
