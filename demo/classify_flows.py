"""
Joins Zeek's conn.log (proto, service, ports) with flowmeter.log (the
derived statistical features) on uid, aligns columns to the trained
model's expected schema, classifies each resulting flow, saves full
per-flow results to CSV, and prints detection metrics against known
ground truth derived from the simulation's destination ports.

Run after run_full_simulation.sh has produced conn.log and flowmeter.log
in the current directory. Expects an artifacts/ directory with:
  random_forest_model.joblib, scaler.joblib, column_transformer.joblib,
  label_encoder.joblib, feature_names.joblib, categorical_cols.joblib,
  numeric_cols.joblib
"""
import os
import sys
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Ground truth for THIS simulation only: maps the destination port used by
# each traffic generator in run_full_simulation.sh to the true attack/benign
# category. Edit this if you change which ports/tools you simulate with.
PORT_TO_TRUE_LABEL = {
    1883: 'MQTT_Publish',      # mosquitto_pub traffic
    80: 'DOS_SYN_Hping',       # hping3 -S -p 80 traffic
}

OUTPUT_CSV = 'classification_results.csv'
MAX_CONSOLE_ROWS = 20  # full detail always goes to CSV regardless of this


def find_artifacts_dir():
    """Looks for the artifacts directory in common locations relative to
    wherever this script is run from, so it works whether you're in the
    repo root or a subfolder like demo/."""
    candidates = ['artifacts', '../artifacts', '../../artifacts']
    override = os.environ.get('ARTIFACTS_DIR')
    if override:
        candidates.insert(0, override)

    for c in candidates:
        if os.path.isfile(os.path.join(c, 'random_forest_model.joblib')):
            print(f"Using artifacts directory: {os.path.abspath(c)}")
            return c

    print("ERROR: could not find artifacts/random_forest_model.joblib in any of:")
    for c in candidates:
        print(f"  - {os.path.abspath(c)}")
    print("Set ARTIFACTS_DIR=/path/to/artifacts and rerun, e.g.:")
    print("  ARTIFACTS_DIR=../artifacts python3 classify_flows.py")
    sys.exit(1)


def read_zeek_log(path):
    """Reads a Zeek TSV log, using the '#fields' comment line as the header."""
    with open(path) as f:
        lines = f.readlines()

    header = None
    for line in lines:
        if line.startswith('#fields'):
            header = line.strip().split('\t')[1:]
            break
    if header is None:
        raise ValueError(f"Could not find #fields header in {path}")

    data_lines = [l for l in lines if not l.startswith('#')]
    from io import StringIO
    # NOTE: deliberately NOT treating '-' as NaN here. Zeek uses '-' for
    # "unset", but RT-IoT2022's 'service' column also legitimately uses the
    # literal string '-' as a real category (source of the 'service_-'
    # one-hot column) — converting it to NaN would silently break service
    # classification. Numeric columns are coerced explicitly later instead.
    df = pd.read_csv(StringIO(''.join(data_lines)), sep='\t', names=header)
    return df


def derive_ground_truth(merged):
    """Maps each flow's destination port to a known true label using
    PORT_TO_TRUE_LABEL. Flows to any other port are labeled 'unknown' and
    excluded from metrics (but still classified and saved to the CSV)."""
    resp_port = pd.to_numeric(merged.get('id.resp_p'), errors='coerce')
    return resp_port.map(PORT_TO_TRUE_LABEL).fillna('unknown')


def main():
    print("Reading conn.log and flowmeter.log...")
    try:
        conn = read_zeek_log('conn.log')
        flowmeter = read_zeek_log('flowmeter.log')
    except Exception as e:
        print(f"ERROR reading logs: {e}")
        print("Run: head -20 conn.log   and   head -20 flowmeter.log")
        print("to check the actual format, then adjust read_zeek_log() if needed.")
        sys.exit(1)

    print(f"conn.log: {len(conn)} rows, flowmeter.log: {len(flowmeter)} rows")

    if 'uid' not in conn.columns or 'uid' not in flowmeter.columns:
        print("ERROR: 'uid' column missing from one of the logs — cannot join.")
        print("conn.log columns:", list(conn.columns))
        print("flowmeter.log columns:", list(flowmeter.columns))
        sys.exit(1)

    conn_cols_needed = [c for c in ['uid', 'proto', 'service', 'id.orig_p', 'id.resp_p'] if c in conn.columns]
    merged = flowmeter.merge(conn[conn_cols_needed], on='uid', how='left')
    print(f"Merged: {len(merged)} rows")

    print("\nLoading saved artifacts...")
    artifacts_dir = find_artifacts_dir()
    try:
        ct = joblib.load(os.path.join(artifacts_dir, 'column_transformer.joblib'))
        scaler = joblib.load(os.path.join(artifacts_dir, 'scaler.joblib'))
        rf = joblib.load(os.path.join(artifacts_dir, 'random_forest_model.joblib'))
        label_encoder = joblib.load(os.path.join(artifacts_dir, 'label_encoder.joblib'))
        categorical_cols = joblib.load(os.path.join(artifacts_dir, 'categorical_cols.joblib'))
        numeric_cols = joblib.load(os.path.join(artifacts_dir, 'numeric_cols.joblib'))
    except FileNotFoundError as e:
        print(f"ERROR: missing artifact file: {e}")
        print("Make sure artifacts/ was produced by train_evaluate(..., save_model=True).")
        sys.exit(1)

    all_expected_cols = list(categorical_cols) + list(numeric_cols)
    missing = [c for c in all_expected_cols if c not in merged.columns]

    print(f"\nExpected columns: {len(all_expected_cols)}")
    print(f"Missing columns: {len(missing)}")
    if missing:
        print("Missing:", missing)
        print("These will be 0-filled — treat results as an approximation, not")
        print("a faithful reproduction of the original RT-IoT2022 feature values.")
        for c in missing:
            merged[c] = 0

    # fill any remaining NaNs (e.g. from unmatched uids or non-TCP flows) with
    # a sensible default per column type. Coercing numeric columns is where
    # Zeek's '-' (unset) is correctly treated as missing/0. Categorical
    # columns (proto, service) get '-' as the fallback, matching the
    # dataset's own convention for "no service classified" — NOT 0, which
    # is an unseen category the encoder would just ignore.
    for c in numeric_cols:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors='coerce')
    merged[list(numeric_cols)] = merged[list(numeric_cols)].fillna(0)
    for c in categorical_cols:
        if c in merged.columns:
            merged[c] = merged[c].fillna('-')

    X_raw = merged[all_expected_cols]
    X_encoded = ct.transform(X_raw)
    X_scaled = scaler.transform(X_encoded)

    preds = rf.predict(X_scaled)
    probas = rf.predict_proba(X_scaled)
    predicted_labels = label_encoder.inverse_transform(preds)
    confidences = probas.max(axis=1)

    true_labels = derive_ground_truth(merged)

    # --- Build the full results table and save it ---
    results = pd.DataFrame({
        'uid': merged['uid'],
        'id.orig_p': merged.get('id.orig_p'),
        'id.resp_p': merged.get('id.resp_p'),
        'proto': merged.get('proto'),
        'service': merged.get('service'),
        'true_label': true_labels,
        'predicted_label': predicted_labels,
        'confidence': confidences,
        'correct': true_labels == predicted_labels,
    })
    results.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved full per-flow results to: {OUTPUT_CSV}")

    # --- Console preview (capped, since captures can have thousands of flows) ---
    print(f"\n=== Classification results (first {min(MAX_CONSOLE_ROWS, len(results))} of {len(results)} flows) ===")
    print(results.head(MAX_CONSOLE_ROWS).to_string(index=False))
    if len(results) > MAX_CONSOLE_ROWS:
        print(f"... ({len(results) - MAX_CONSOLE_ROWS} more rows in {OUTPUT_CSV})")

    print("\nPredicted class distribution (all flows):")
    print(predicted_labels_series := pd.Series(predicted_labels).value_counts())

    # --- Detection metrics against known ground truth ---
    known = results[results['true_label'] != 'unknown']
    unknown_count = len(results) - len(known)

    print(f"\n=== Detection metrics (flows with known ground truth: {len(known)}/{len(results)}) ===")
    if unknown_count:
        print(f"({unknown_count} flows had a destination port not in PORT_TO_TRUE_LABEL "
              f"and were excluded from metrics — see '{OUTPUT_CSV}' for their predictions)")

    if len(known) == 0:
        print("No flows matched a known port in PORT_TO_TRUE_LABEL — cannot compute metrics.")
        print("Check that PORT_TO_TRUE_LABEL matches the ports you actually simulated.")
        return

    y_true = known['true_label']
    y_pred = known['predicted_label']

    acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall accuracy (known-label flows only): {acc:.2%}")

    print("\nPer-class report:")
    known_classes = sorted(y_true.unique())
    print(classification_report(y_true, y_pred, labels=known_classes, zero_division=0))

    print("Confusion matrix (rows = true, columns = predicted; "
          "columns beyond the known classes indicate the model predicted a class we didn't simulate):")
    all_seen_labels = sorted(set(y_true.unique()) | set(y_pred.unique()))
    cm = confusion_matrix(y_true, y_pred, labels=all_seen_labels)
    cm_df = pd.DataFrame(cm, index=all_seen_labels, columns=all_seen_labels)
    print(cm_df)

    print("\nPer-category accuracy:")
    for label in known_classes:
        subset = known[known['true_label'] == label]
        cat_acc = (subset['predicted_label'] == label).mean()
        print(f"  {label}: {cat_acc:.2%} ({subset['correct'].sum()}/{len(subset)} correctly detected)")


if __name__ == '__main__':
    main()