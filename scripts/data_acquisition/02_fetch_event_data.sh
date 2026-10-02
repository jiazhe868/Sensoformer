#!/bin/bash
# Step 2: For every event in the cleaned list, fetch (a) the phase-pick file
# ({evid}.dat) and (b) the triggered waveforms as per-event SAC directories,
# both via STP. Waveform downloads run in parallel batches.
#
# STP's TRIG writes {evid}/{evid}.{NET}.{STA}.{CHA}.sac with event/station
# geometry headers (evla, evlo, evdp, stla, stlo, dist, az) prefilled, and the
# automatic first-arrival marker in header 'a' where available.
#
# Usage: ./02_fetch_event_data.sh [EVENT_LIST] [RADIUS_KM] [CHANNELS] [MAX_JOBS] [DELAY_S]
set -e

INPUT_FILE=${1:-events_cleaned.dat}
RADIUS=${2:-200}
CHANNELS=${3:-"HH_ EH_"}
MAX_JOBS=${4:-20}
DELAY=${5:-0.5}

[ -f "$INPUT_FILE" ] || { echo "Event list '$INPUT_FILE' not found" >&2; exit 1; }

# (a) Phase-pick files for all events in one STP session
gawk '{print "PHASE -f "$1".dat -e "$1}' "$INPUT_FILE" > stp.script.phase
stp << STP_EOF
IN stp.script.phase
STP_EOF

# (b) Waveforms, parallel in batches
TOTAL=$(wc -l < "$INPUT_FILE")
COUNT=0; IN_ROUND=0
download_event() {
    stp >/dev/null 2>&1 << STP_EOF
GAIN ON
TRIG -radius ${RADIUS} -chan ${CHANNELS} $1
STP_EOF
}
echo "Downloading waveforms for ${TOTAL} events (batches of ${MAX_JOBS})..."
while read -r event_id _; do
    download_event "$event_id" &
    IN_ROUND=$((IN_ROUND + 1)); COUNT=$((COUNT + 1))
    printf "\rprogress: %d/%d" "$COUNT" "$TOTAL"
    sleep "$DELAY"
    if [ "$IN_ROUND" -ge "$MAX_JOBS" ]; then
        wait; IN_ROUND=0
    fi
done < "$INPUT_FILE"
wait
echo; echo "Done."
