#!/usr/bin/env bash
# ============================================================================
# Full simulation pipeline: install Zeek + flowmeter, generate 2 categories
# of traffic, capture, extract flow features, and classify with the trained
# RT-IoT2022 model.
#
# Run from the directory containing your `artifacts/` folder
# (random_forest_model.joblib, scaler.joblib, column_transformer.joblib,
#  label_encoder.joblib, feature_names.joblib, categorical_cols.joblib,
#  numeric_cols.joblib).
#
# Usage: sudo bash run_full_simulation.sh
#   (needs sudo for tcpdump/hping3/apt; run as your normal user with sudo
#    available, not as root directly, so mosquitto_pub etc. still work)
# ============================================================================
set -e

CAPTURE_IFACE="lo"
CAPTURE_FILE="capture.pcap"
CAPTURE_SECONDS=15
MQTT_MESSAGES=50
HPING_COUNT=2000

fail() {
    echo ""
    echo "!!! FAILED at: $1"
    echo "!!! If you're short on time, abandon this script now and use the"
    echo "!!! two-test-set-flows fallback script instead."
    exit 1
}

echo "=== [1/6] Detecting Debian version ==="
DEBIAN_VER=$(cat /etc/debian_version 2>/dev/null | cut -d. -f1)
if [ -z "$DEBIAN_VER" ]; then
    echo "Could not detect Debian version, defaulting to Debian_12"
    ZEEK_REPO="Debian_12"
elif [ "$DEBIAN_VER" = "12" ]; then
    ZEEK_REPO="Debian_12"
elif [ "$DEBIAN_VER" = "13" ]; then
    ZEEK_REPO="Debian_13"
else
    echo "Debian version '$DEBIAN_VER' not directly mapped, trying Debian_Testing"
    ZEEK_REPO="Debian_Testing"
fi
echo "Using Zeek repo: $ZEEK_REPO"

echo "=== [2/6] Installing Zeek (skipped if already installed) ==="
command -v curl >/dev/null 2>&1 || sudo apt install -y curl || fail "installing curl"
if ! command -v zeek >/dev/null 2>&1 && [ ! -x /opt/zeek/bin/zeek ]; then
    curl -fsSL "https://download.opensuse.org/repositories/security:zeek/${ZEEK_REPO}/Release.key" \
        | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/security_zeek.gpg > /dev/null \
        || fail "adding Zeek GPG key"
    [ -s /etc/apt/trusted.gpg.d/security_zeek.gpg ] || fail "Zeek GPG key file is empty — check curl/network, then rerun"
    echo "deb http://download.opensuse.org/repositories/security:/zeek/${ZEEK_REPO}/ /" \
        | sudo tee /etc/apt/sources.list.d/security:zeek.list > /dev/null
    sudo apt update || fail "apt update"
    sudo apt install -y zeek || fail "installing zeek"
else
    echo "Zeek already installed, skipping"
fi
export PATH="/opt/zeek/bin:$PATH"
zeek --version || fail "zeek not on PATH after install"

echo "=== [3/6] Installing flowmeter plugin + traffic tools ==="
if [ ! -d "/opt/zeek/share/zeek/site/flowmeter" ] && [ ! -d "zeek-flowmeter" ]; then
    sudo apt install -y git hping3 mosquitto mosquitto-clients tcpdump || fail "installing tools"
    git clone https://github.com/zeek-flowmeter/zeek-flowmeter.git || fail "cloning flowmeter plugin"
    (cd zeek-flowmeter && zkg install . --force) || {
        echo "zkg install failed, falling back to manual copy"
        sudo mkdir -p /opt/zeek/share/zeek/site/flowmeter
        sudo cp -r zeek-flowmeter/scripts/* /opt/zeek/share/zeek/site/flowmeter/
    }
else
    sudo apt install -y hping3 mosquitto mosquitto-clients tcpdump 2>/dev/null || true
    echo "flowmeter plugin already present, skipping install"
fi

sudo systemctl start mosquitto 2>/dev/null || sudo mosquitto -d || echo "mosquitto may already be running"

echo "=== [4/6] Capturing traffic (category 1: MQTT_Publish, category 2: DOS_SYN_Hping) ==="
rm -f "$CAPTURE_FILE"
sudo timeout "$CAPTURE_SECONDS" tcpdump -i "$CAPTURE_IFACE" -w "$CAPTURE_FILE" &
TCPDUMP_PID=$!
sleep 1   # let tcpdump attach before traffic starts

echo "  -> generating MQTT_Publish traffic ($MQTT_MESSAGES messages)"
for i in $(seq 1 "$MQTT_MESSAGES"); do
    mosquitto_pub -h localhost -t test/topic -m "reading_$i" 2>/dev/null
done

echo "  -> generating DOS_SYN_Hping traffic ($HPING_COUNT packets)"
sudo hping3 -S -p 80 --faster -c "$HPING_COUNT" localhost 2>/dev/null || true

wait "$TCPDUMP_PID" 2>/dev/null || true
echo "  -> capture saved to $CAPTURE_FILE"

echo "=== [5/6] Running Zeek + flowmeter on the capture ==="
[ -s "$CAPTURE_FILE" ] || fail "capture file is empty — check tcpdump permissions/interface name"
rm -f conn.log flowmeter.log
zeek -C -r "$CAPTURE_FILE" flowmeter || fail "zeek analysis"
[ -f conn.log ] && [ -f flowmeter.log ] || fail "conn.log or flowmeter.log was not produced"
echo "  -> conn.log and flowmeter.log generated"

echo "=== [6/6] Joining logs and classifying with the trained model ==="
python3 classify_flows.py || fail "classification step (see classify_flows.py output above)"

echo ""
echo "=== DONE ==="