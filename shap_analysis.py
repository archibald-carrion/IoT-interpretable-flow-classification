import matplotlib.pyplot as plt
import matplotlib
import shap

def shap_analysis(results, X_train_scaled, feature_names, label_encoder):
    """Run SHAP analysis for the trained Random Forest model."""
    print("\n--- SHAP Analysis for Random Forest ---")
    model = results['model']

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train_scaled[:1000])

    n_features = min(20, len(feature_names))
    n_classes = len(label_encoder.classes_)

    # --- Plot 1: Bar Chart (Feature Importance) ---
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
    plt.savefig('shap_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: shap_feature_importance.png")

    # --- Plot 2: Dot Summary Plot ---
    fig_height = max(12, n_features * 0.55 + 3)
    fig, ax = plt.subplots(figsize=(18, fig_height))

    shap.summary_plot(
        shap_values,
        X_train_scaled[:1000],
        feature_names=feature_names,
        class_names=label_encoder.classes_,
        show=False,
        max_display=n_features,
    )

    ax = plt.gca()
    ax.set_title('SHAP Summary Plot - Random Forest', fontsize=18, pad=15)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_xlabel(ax.get_xlabel(), fontsize=13)

    legend = ax.get_legend()
    if legend:
        legend.set_bbox_to_anchor((1.02, 1))
        legend.set_loc('upper left')
        for text in legend.get_texts():
            text.set_fontsize(11)

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.savefig('shap_summary_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: shap_summary_plot.png")