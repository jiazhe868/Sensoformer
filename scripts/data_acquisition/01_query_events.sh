#!/bin/bash
# Step 1: Query the SCEDC event catalog via STP and write events.dat /
# events_cleaned.dat.
#
# Requires the STP client (https://scedc.caltech.edu/data/stp/) on PATH and
# network access to SCEDC.
#
# Usage (defaults reproduce the released training catalog selection):
#   ./01_query_events.sh [MINMAG MAXMAG T0 T1 MINDEP MAXDEP MINLAT MAXLAT MINLON MAXLON]
# Example:
#   ./01_query_events.sh 3 8 19900101000000 20250301000000 0 30.0 31 39 -122 -114
set -e

MINMAG=${1:-3}
MAXMAG=${2:-8}
T0=${3:-19900101000000}
T1=${4:-20250301000000}
MINDEP=${5:-0}
MAXDEP=${6:-30.0}
MINLAT=${7:-31}
MAXLAT=${8:-39}
MINLON=${9:--122}
MAXLON=${10:--114}

stp << STP_EOF > events.dat
SET NEVNTMAX 99999
EVENT -mag ${MINMAG} ${MAXMAG} -t0 ${T0} ${T1} -depth ${MINDEP} ${MAXDEP} -lat ${MINLAT} ${MAXLAT} -lon ${MINLON} ${MAXLON}
STP_EOF

# Remove STP protocol noise (lines containing "return") -> cleaned list.
grep -v "return" events.dat > events_cleaned.dat
echo "Wrote $(wc -l < events_cleaned.dat) events to events_cleaned.dat"
