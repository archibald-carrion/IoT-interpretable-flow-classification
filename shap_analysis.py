import numpy as np
import matplotlib.pyplot as plt
import shap

def shap_analysis(results, X_train_scaled, y_train, feature_names, label_encoder,
                   n_samples_per_class=100, normalize='within_class'):
    """
    SHAP analysis with class-balanced sampling so importance isn't
    dominated by majority classes.

    normalize:
        'within_class' -> each class's segment lengths sum to 1
                           (mirrors your confusion-matrix row-normalization:
                            "of this class's total SHAP mass, how much
                            comes from each feature")
        None            -> raw mean(|SHAP|), balanced sampling only
    """
    print("\n--- SHAP Analysis for Random Forest ---")
    model = results['model']
    n_classes = len(label_encoder.classes_)
    n_features = min(20, len(feature_names))

    # --- 1. Class-balanced sample (this is the actual fix) ---
    rng = np.random.default_rng(42)
    idx_parts = []
    for c in range(n_classes):
        class_idx = np.where(y_train == c)[0]
        if len(class_idx) == 0:
            continue
        take = min(n_samples_per_class, len(class_idx))
        idx_parts.append(rng.choice(class_idx, size=take, replace=False))
    sample_idx = np.concatenate(idx_parts)
    X_sample = X_train_scaled[sample_idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # --- 2. Build importance matrix: (n_features, n_classes) ---
    # SHAP's TreeExplainer output format for multiclass differs by version:
    #   older shap: list of n_classes arrays, each (n_samples, n_features)
    #   newer shap: single array, shape (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        importance = np.stack([np.abs(sv).mean(axis=0) for sv in shap_values], axis=1)
    else:
        # shap_values.shape == (n_samples, n_features, n_classes)
        importance = np.abs(shap_values).mean(axis=0)  # -> (n_features, n_classes) directly

    print(f"shap_values type: {'list' if isinstance(shap_values, list) else 'ndarray'}, "
          f"importance matrix shape: {importance.shape} "
          f"(expected: ({len(feature_names)}, {n_classes}))")

    if normalize == 'within_class':
        col_sums = importance.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1  # avoid div-by-zero
        importance = importance / col_sums

    # --- 3. Pick top features by total importance, sort ascending for barh ---
    total_importance = importance.sum(axis=1)
    top_idx = np.argsort(total_importance)[-n_features:]
    importance_top = importance[top_idx]
    feature_names_top = [feature_names[i] for i in top_idx]

    # --- 4. Custom stacked horizontal bar chart (full control, no shap magic) ---
    fig_height = max(10, n_features * 0.5 + n_classes * 0.4)
    fig, ax = plt.subplots(figsize=(16, fig_height))

    colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
    left = np.zeros(n_features)
    for c in range(n_classes):
        ax.barh(feature_names_top, importance_top[:, c], left=left,
                color=colors[c], label=label_encoder.classes_[c])
        left += importance_top[:, c]

    ax.set_title('SHAP Feature Importance - Random Forest (class-balanced)',
                  fontsize=18, pad=15)
    xlabel = ('Share of class SHAP mass' if normalize == 'within_class'
              else 'mean(|SHAP value|)')
    ax.set_xlabel(xlabel, fontsize=13)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.savefig('shap_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: shap_feature_importance.png")

    # --- Dot summary plot: also use the balanced sample ---
    fig_height = max(12, n_features * 0.55 + 3)
    fig, ax = plt.subplots(figsize=(18, fig_height))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        class_names=label_encoder.classes_,
        show=False,
        max_display=n_features,
    )
    ax = plt.gca()
    ax.set_title('SHAP Summary Plot - Random Forest (class-balanced)', fontsize=18, pad=15)
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.savefig('shap_summary_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: shap_summary_plot.png")

    return shap_values, X_sample