#!/usr/bin/env bash
# attach/test_runtime_attach.sh — contract test for Project 7 attach interface
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATTACH="$SCRIPT_DIR/runtime_attach.sh"

TMP="$(mktemp -d /tmp/p7_test_attach_XXXXXX)"
trap 'rm -rf "$TMP"' EXIT

# Create a mock INT8 pin
mkdir -p "$TMP/int8_pin"
cat > "$TMP/int8_pin/pin.json" <<EOF
{"schema":"bf16_to_int8_pin_v1"}
EOF

# Create a mock nested-nf8 pin
mkdir -p "$TMP/nested_pin"
cat > "$TMP/nested_pin/pin.json" <<EOF
{"schema":"nested_nf8_pin_v1"}
EOF

echo "--- Test 1: Help flag prints usage and exits 0 ---"
set +e
"$ATTACH" -h >/dev/null
RC=$?
set -e
[[ $RC -eq 0 ]] || { echo "FAIL: -h did not exit 0, got $RC"; exit 1; }

echo "--- Test 2: Missing args exits 2 ---"
set +e
"$ATTACH" --phase train >/dev/null 2>&1
RC=$?
set -e
[[ $RC -eq 2 ]] || { echo "FAIL: missing args did not exit 2, got $RC"; exit 1; }

echo "--- Test 3: Unknown flag exits 2 ---"
set +e
"$ATTACH" --phase train --out-dir "$TMP/out" --mode 1 --pin "$TMP/int8_pin" --bogus >/dev/null 2>&1
RC=$?
set -e
[[ $RC -eq 2 ]] || { echo "FAIL: unknown flag did not exit 2, got $RC"; exit 1; }

echo "--- Test 4: Mode outside 1..5 exits 2 ---"
set +e
"$ATTACH" --phase train --out-dir "$TMP/out" --mode 6 --pin "$TMP/int8_pin" >/dev/null 2>&1
RC=$?
set -e
[[ $RC -eq 2 ]] || { echo "FAIL: mode 6 did not exit 2, got $RC"; exit 1; }

echo "--- Test 5: Missing pin path exits 4 (HARD_BLOCK) ---"
set +e
"$ATTACH" --phase train --out-dir "$TMP/out" --mode 1 --pin "$TMP/nonexistent" >/dev/null 2>&1
RC=$?
set -e
[[ $RC -eq 4 ]] || { echo "FAIL: missing pin did not exit 4, got $RC"; exit 1; }

echo "--- Test 6: Schema mismatch exits 4 (HARD_BLOCK) ---"
set +e
# Mode 3 requires nested_nf8_pin_v1, passing int8_pin should fail
"$ATTACH" --phase train --out-dir "$TMP/out" --mode 3 --pin "$TMP/int8_pin" >/dev/null 2>&1
RC=$?
set -e
[[ $RC -eq 4 ]] || { echo "FAIL: schema mismatch did not exit 4, got $RC"; exit 1; }

echo "--- Test 7: Valid Mode 1 attach exits 0 and writes env ---"
OUT1="$TMP/out1"
"$ATTACH" --phase train --out-dir "$OUT1" --mode 1 --pin "$TMP/int8_pin" >/dev/null
[[ -f "$OUT1/boards/p7_runtime_attach.env" ]] || { echo "FAIL: env file missing"; exit 1; }
[[ -f "$OUT1/boards/p7_runtime_attach_v1.json" ]] || { echo "FAIL: stamp file missing"; exit 1; }
grep -q "export P7_ATTACHED=1" "$OUT1/boards/p7_runtime_attach.env" || { echo "FAIL: P7_ATTACHED not set"; exit 1; }
grep -q '"status": "attached"' "$OUT1/boards/p7_runtime_attach_v1.json" || { echo "FAIL: status not attached"; exit 1; }

echo "--- Test 8: Valid Mode 3 attach exits 0 and writes env ---"
OUT3="$TMP/out3"
"$ATTACH" --phase train --out-dir "$OUT3" --mode 3 --pin "$TMP/nested_pin" >/dev/null
[[ -f "$OUT3/boards/p7_runtime_attach.env" ]] || { echo "FAIL: env file missing"; exit 1; }
[[ -f "$OUT3/boards/p7_runtime_attach_v1.json" ]] || { echo "FAIL: stamp file missing"; exit 1; }
grep -q "export P7_MODE=\"3\"" "$OUT3/boards/p7_runtime_attach.env" || { echo "FAIL: P7_MODE not 3"; exit 1; }

echo "TEST_RUNTIME_ATTACH_PASS: all 8 contract checks passed"
exit 0
