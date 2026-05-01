#!/usr/bin/env bash
set -uo pipefail

# ════════════════════════════════════════════════════════════════
# IPTV-ORG EPG FAST MULTI-SITE XML GRABBER
# ════════════════════════════════════════════════════════════════

SCRIPT_START_TIME=$(date +%s)

# ── Config ──────────────────────────────────────────────────────
REPO_URL="https://github.com/iptv-org/epg"

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$BASE_DIR/epg"
OUTPUT_DIR="$BASE_DIR/sites"
LOG_DIR="$OUTPUT_DIR/logs"

GENERATE_CONTENT_SCRIPT="$BASE_DIR/content_generator.py"
XML_SPLITTER_SCRIPT="$BASE_DIR/xml_splitter.py"
XML_FORMATTER_SCRIPT="$BASE_DIR/xml_formatter.py"
CONTENT_JSON="$OUTPUT_DIR/content.json"
SITES_MD="$WORK_DIR/SITES.md"

PROXY_URL="${PROXY_URL:-}"

DELAY="${DELAY:-0}"
TIMEOUT="${TIMEOUT:-15000}"
MAX_CONN="${MAX_CONN:-50}"
MIN_CONN="${MIN_CONN:-1}"
MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_BACKOFF_BASE="${RETRY_BACKOFF_BASE:-2}"
BATCH_SIZE="${BATCH_SIZE:-10}"

# ── Workers ─────────────────────────────────────────────────────
detect_workers() {
  local cpus mem_gb
  cpus=$(nproc 2>/dev/null || echo 2)
  mem_gb=$(awk '/MemTotal/{printf "%d",$2/1024/1024}' /proc/meminfo 2>/dev/null || echo 4)

  local w=$(( cpus * 3 < mem_gb * 3 / 2 ? cpus * 3 : mem_gb * 3 / 2 ))
  (( w < 1 )) && w=1
  (( w > 32 )) && w=32
  echo "$w"
}

PARALLEL="${PARALLEL:-$(detect_workers)}"

# ── Colors ──────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'
CYAN='\033[0;36m'; MAGENTA='\033[0;35m'; BOLD='\033[1m'
DIM='\033[2m'; NC='\033[0m'

log(){ echo -e "${GREEN}[INFO]${NC} $*"; }
warn(){ echo -e "${YELLOW}[WARN]${NC} $*"; }
err(){ echo -e "${RED}[ERR]${NC} $*" >&2; }

elapsed_since(){
  local s=$(( $(date +%s) - $1 ))
  printf "%dm%02ds" $((s/60)) $((s%60))
}

# ── Cleanup (FIXED TRAP MERGE) ──────────────────────────────────
WORKER_SCRIPT=""
BATCH_ARG_FILE=""

cleanup(){
  [[ "${KEEP_REPO:-0}" != "1" && -d "$WORK_DIR" ]] && rm -rf "$WORK_DIR"
  [[ -n "$WORKER_SCRIPT" && -f "$WORKER_SCRIPT" ]] && rm -f "$WORKER_SCRIPT"
  [[ -n "$BATCH_ARG_FILE" && -f "$BATCH_ARG_FILE" ]] && rm -f "$BATCH_ARG_FILE"
}
trap cleanup EXIT

# ── Dependencies ───────────────────────────────────────────────
for cmd in git npm python3 grep sed sort wc xargs awk; do
  command -v "$cmd" >/dev/null || { err "Missing: $cmd"; exit 1; }
done

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# ── Repo ────────────────────────────────────────────────────────
if [[ -d "$WORK_DIR/.git" ]]; then
  git -C "$WORK_DIR" fetch --depth 1 origin master >/dev/null 2>&1 || true
  git -C "$WORK_DIR" reset --hard origin/master >/dev/null 2>&1 || true
else
  git clone --depth 1 "$REPO_URL" "$WORK_DIR"
fi

cd "$WORK_DIR"
npm ci --silent

# ── Sites ───────────────────────────────────────────────────────
mapfile -t ONLINE_SITES < <(
  grep '🟢' "$SITES_MD" | sed -n 's#.*href="sites/\([^"]*\)".*🟢.*#\1#p' | sort -u
)

SITES=()
for s in "${ONLINE_SITES[@]}"; do
  [[ -d "$WORK_DIR/sites/$s" ]] && SITES+=("$s")
done

TOTAL="${#SITES[@]}"
(( TOTAL == 0 )) && { err "No sites"; exit 1; }

log "Sites: $TOTAL | Workers: $PARALLEL"

# ── Worker script ───────────────────────────────────────────────
WORKER_SCRIPT=$(mktemp /tmp/epg_worker_XXXXXX.sh)
chmod +x "$WORKER_SCRIPT"

cat > "$WORKER_SCRIPT" <<'EOF'
#!/usr/bin/env bash
trap '' PIPE

run_batch(){
  local sites=("$@")

  for s in "${sites[@]}"; do
    echo "WORKER_START|$s"
  done

  npm run grab -- \
    --sites="$(IFS=,; echo "${sites[*]}")" \
    --output="sites/{site}.xml" \
    --delay="$DELAY" \
    --timeout="$TIMEOUT" \
    --maxConnections="$MAX_CONN" \
    ${PROXY_URL:+--proxy="$PROXY_URL"} \
    > /tmp/epg_batch.log 2>&1 || true

  for s in "${sites[@]}"; do
    if [[ -f "sites/$s.xml" ]]; then
      echo "PASS|$s"
    else
      echo "FAIL|$s"
    fi
  done
}

run_batch "$@"
EOF

# ── Batch file ────────────────────────────────────────────────
BATCH_ARG_FILE=$(mktemp)

count=0
line=""

for s in "${SITES[@]}"; do
  line+="$s "
  ((count++))
  ((count % BATCH_SIZE == 0)) && { echo "${line% }" >> "$BATCH_ARG_FILE"; line=""; }
done
[[ -n "$line" ]] && echo "${line% }" >> "$BATCH_ARG_FILE"

# ── CRITICAL FIX: disable pipefail for aggregator ──────────────
set +o pipefail

PASS=0; FAIL=0

while IFS='|' read -r token a b c; do
  case "$token" in
    WORKER_START)
      echo -e "${DIM}[START] $a${NC}"
      ;;
    PASS)
      ((PASS++))
      echo -e "${GREEN}[OK] $a${NC}"
      ;;
    FAIL)
      ((FAIL++))
      echo -e "${RED}[FAIL] $a${NC}"
      ;;
  esac
done < <(
  xargs -L1 -P "$PARALLEL" bash "$WORKER_SCRIPT" < "$BATCH_ARG_FILE" || true
)

set -o pipefail

# ── Python steps ───────────────────────────────────────────────
log "Splitting XML..."
python3 "$XML_SPLITTER_SCRIPT" "$OUTPUT_DIR"

log "Formatting XML..."
python3 "$XML_FORMATTER_SCRIPT"

log "Generating content.json..."
python3 "$GENERATE_CONTENT_SCRIPT" "$OUTPUT_DIR" "$CONTENT_JSON"

# ── Summary ────────────────────────────────────────────────────
echo ""
echo "PASS: $PASS"
echo "FAIL: $FAIL"
echo "DONE"
