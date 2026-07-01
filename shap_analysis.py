import numpy as np
import matplotlib.pyplot as plt
import shap

def plot_shap_dot_summary(shap_values, y_sample, feature_names, class_names, max_display=20):
    """
    Custom dot summary plot: one dot per class per feature.
    - x-axis: Mean SHAP value, computed only over samples whose TRUE label
      is that class (class-conditioned mean) — NOT averaged across all classes.
    - color: Unique color per class (from tab20 colormap)
    - size: |mean SHAP| for that specific (feature, class) pair
    - legend: Maps colors to class names
    """
    # Convert shap_values to (n_samples, n_features, n_classes) if needed
    if isinstance(shap_values, list):
        shap_values = np.stack(shap_values, axis=-1)  # (n_samples, n_features, n_classes)

    n_samples, n_features, n_classes = shap_values.shape

    # --- Class-conditioned mean SHAP: average only over each class's own samples ---
    # (averaging over ALL balanced samples instead of just the class's own samples
    #  dilutes/flips the sign for features that are rare-but-strong indicators,
    #  e.g. one-hot service flags)
    mean_shap = np.zeros((n_features, n_classes))
    for c in range(n_classes):
        mask = (y_sample == c)
        if mask.sum() == 0:
            continue
        mean_shap[:, c] = shap_values[mask, :, c].mean(axis=0)

    # Get top features by total conditioned |SHAP| across classes
    feature_order = np.argsort(np.sum(np.abs(mean_shap), axis=1))[::-1][:max_display]
    mean_shap = mean_shap[feature_order]
    feature_names = [feature_names[i] for i in feature_order]

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Use a colormap with enough distinct colors for all classes
    colors = plt.cm.tab20(np.linspace(0, 1, n_classes))

    # Per-(feature, class) dot size — NOT collapsed across classes anymore
    sizes_matrix = np.abs(mean_shap)
    sizes_matrix = 100 * (sizes_matrix / sizes_matrix.max()) + 10  # Scale to [10, 110]

    # Plot one dot per class per feature
    for j, class_name in enumerate(class_names):
        for i, feature in enumerate(feature_names):
            shap_val = mean_shap[i, j]
            ax.scatter(
                shap_val, i,
                s=sizes_matrix[i, j],
                c=[colors[j]],  # Unique color per class
                alpha=0.8,
                edgecolors='black',
                linewidth=0.5,
                label=class_name if i == 0 else None  # Only label once per class
            )

    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names)
    ax.set_xlabel('Mean SHAP value, within true class (impact on model output)')
    ax.set_title('SHAP Dot Summary: One Dot = One Class\n(Size = |mean SHAP| for that class, Color = Class)')
    ax.axvline(0, color='black', linestyle='--', alpha=0.3)

    # Add legend outside the plot
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=10,
        title='Class'
    )

    plt.tight_layout()
    plt.savefig('shap_dot_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

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
    y_sample_parts = []
    for c in range(n_classes):
        class_idx = np.where(y_train == c)[0]
        if len(class_idx) == 0:
            continue
        take = min(n_samples_per_class, len(class_idx))
        chosen = rng.choice(class_idx, size=take, replace=False)
        idx_parts.append(chosen)
        y_sample_parts.append(np.full(take, c))
    sample_idx = np.concatenate(idx_parts)
    y_sample = np.concatenate(y_sample_parts)  # true class label per row of X_sample
    X_sample = X_train_scaled[sample_idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)









    # --- OLD Plot 1: Bar Chart (Feature Importance) only used for comparison with the new one ---
    fig_height = max(10, n_features * 0.5 + n_classes * 0.4)
    fig, ax = plt.subplots(figsize=(16, fig_height))

    shap.summary_plot(
        shap_values,
        X_train_scaled[:1000],
        feature_names=feature_names,
        class_names=label_encoder.classes_,
        plot_type='bar',
        show=False,
        max_display=n_features,
    )

    ax = plt.gca()
    ax.set_title('SHAP Feature Importance - Random Forest', fontsize=18, pad=15)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_xlabel(ax.get_xlabel(), fontsize=13)

    # Fix legend overlap if present
    legend = ax.get_legend()
    if legend:
        legend.set_bbox_to_anchor((1.02, 1))
        legend.set_loc('upper left')
        for text in legend.get_texts():
            text.set_fontsize(11)

    plt.tight_layout(rect=[0, 0, 0.88, 1])  # leave room for legend
    plt.savefig('non_normalized_shap_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: non_normalized_shap_feature_importance.png")








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
    plt.savefig('normalized_shap_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: normalized_shap_feature_importance.png")

    # --- Dot summary plot: also use the balanced sample ---
    plot_shap_dot_summary(shap_values, y_sample, feature_names, label_encoder.classes_)
    print("Saved: shap_dot_summary.png")

    return shap_values, X_sample