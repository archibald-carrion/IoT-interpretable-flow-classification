import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
from joblib import dump


def train_evaluate(X, y, feature_names, label_encoder,
                    column_transformer=None, categorical_cols=None, numeric_cols=None,
                    save_model=False, output_dir='artifacts'):
    """Train a Random Forest classifier and evaluate on a held-out test set.

    Args:
        column_transformer: The fitted ColumnTransformer from load_preprocess_data
            (needed later to one-hot encode new raw flows the same way).
        categorical_cols, numeric_cols: Column lists from load_preprocess_data
            (needed to know which raw fields go where when building a new flow's row).
        save_model: If True, saves every artifact needed for later inference to `output_dir`.
        output_dir: Directory where all .joblib artifacts are written.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)

    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'macro_f1': f1_score(y_test, y_pred, average='macro'),
        'classification_report': classification_report(y_test, y_pred, target_names=label_encoder.classes_),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'model': rf,
        'scaler': scaler,
    }

    print("\n--- Random Forest Performance ---")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Macro F1-Score: {results['macro_f1']:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    print("\nUnique classes in y_test:", np.unique(y_test))
    print("Unique classes in y_pred:", np.unique(y_pred))

    # --- Save every artifact needed to preprocess and classify NEW raw flows later ---
    if save_model:
        os.makedirs(output_dir, exist_ok=True)

        dump(rf, os.path.join(output_dir, 'random_forest_model.joblib'))
        dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
        dump(label_encoder, os.path.join(output_dir, 'label_encoder.joblib'))
        dump(list(feature_names), os.path.join(output_dir, 'feature_names.joblib'))

        saved_files = [
            'random_forest_model.joblib',
            'scaler.joblib',
            'label_encoder.joblib',
            'feature_names.joblib',
        ]

        # These three come from data_processing.py's load_preprocess_data() —
        # required to encode a brand-new raw flow the same way training data was encoded.
        if column_transformer is not None:
            dump(column_transformer, os.path.join(output_dir, 'column_transformer.joblib'))
            saved_files.append('column_transformer.joblib')
        else:
            print("\nWARNING: column_transformer was not provided — new raw flows "
                  "cannot be one-hot encoded consistently without it. Pass the `ct` "
                  "returned by load_preprocess_data() to train_evaluate().")

        if categorical_cols is not None:
            dump(list(categorical_cols), os.path.join(output_dir, 'categorical_cols.joblib'))
            saved_files.append('categorical_cols.joblib')

        if numeric_cols is not None:
            dump(list(numeric_cols), os.path.join(output_dir, 'numeric_cols.joblib'))
            saved_files.append('numeric_cols.joblib')

        print(f"\nSaved {len(saved_files)} inference artifacts to '{output_dir}/':")
        for fname in saved_files:
            print(f"  - {output_dir}/{fname}")

    # --- Confusion Matrix ---
    all_classes = label_encoder.classes_
    n_classes = len(all_classes)
    conf_matrix = confusion_matrix(y_test, y_pred, labels=range(n_classes))

    # Normalize by row (true class totals) for coloring — avoid divide-by-zero
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    conf_matrix_norm = np.divide(
        conf_matrix.astype(float),
        row_sums,
        out=np.zeros_like(conf_matrix, dtype=float),
        where=row_sums != 0
    )

    cell_size = 1.0
    fig_dim = n_classes * cell_size
    fig, ax = plt.subplots(figsize=(fig_dim, fig_dim))

    # Plot normalized matrix for COLORS
    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_matrix_norm,
        display_labels=all_classes
    )
    disp.plot(
        ax=ax,
        cmap='Blues',
        xticks_rotation='vertical',
        colorbar=True,
        values_format='.0%',   # show "98%" etc. on cells temporarily
    )

    # Overwrite cell text with RAW counts (keep proportional coloring)
    flat_counts = conf_matrix.flatten()
    for text, count in zip(ax.texts, flat_counts):
        text.set_text(str(count))

    base_font = fig_dim * 0.9
    cell_font = fig_dim * 0.75

    for text in ax.texts:
        text.set_fontsize(cell_font)

    ax.tick_params(axis='x', labelsize=base_font, rotation=90)
    ax.tick_params(axis='y', labelsize=base_font)

    # Update colorbar label to clarify what the color means
    ax.images[0].colorbar.set_label('Proportion of true class', fontsize=base_font)

    ax.set_title("Confusion Matrix - Random Forest", fontsize=base_font * 1.5, pad=fig_dim * 1.5)
    ax.set_xlabel("Predicted label", fontsize=base_font * 1.2, labelpad=fig_dim)
    ax.set_ylabel("True label", fontsize=base_font * 1.2, labelpad=fig_dim)

    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: confusion_matrix.png")

    return results, X_train_scaled, X_test_scaled, y_train