#!/bin/bash
# Step 3: Write analyst phase picks from the per-event {evid}.dat files into
# the SAC headers the preprocessing expects: t1 = P arrival (on vertical
# component files), t2 = S arrival (on horizontal E/N/1/2 component files).
# Times are seconds relative to the event origin, matching STP's reference.
#
# Requires the SAC binary and gawk. Where analyst picks are unavailable, the
# preprocessing falls back to the automatic 'a' marker written by STP, so
# running this step is recommended but not strictly required.
#
# Usage: ./03_add_picks_to_sac.sh {evid}.dat [more .dat files...]
#        for f in [0-9]*.dat; do ./03_add_picks_to_sac.sh "$f"; done
set -e

for PHASE_FILE in "$@"; do
    [ -f "$PHASE_FILE" ] || { echo "skip: $PHASE_FILE not found"; continue; }
    gawk '{
        if (NR==1) {nm=$1}
        if (NR>1) {
            if ($8=="P" || $8=="p") {
                print "r "nm"/"nm"."$1"."$2".*"substr($3, length($3), 1)".sac";
                print "ch leven true"; print "ch t1 "$13; print "wh";
            } else {
                c = substr($3, length($3), 1);
                if (c=="E" || c=="N") {
                    print "r "nm"/"nm"."$1"."$2".*[EN].sac";
                    print "ch leven true"; print "ch t2 "$13; print "wh";
                }
                if (c=="1" || c=="2") {
                    print "r "nm"/"nm"."$1"."$2".*[12].sac";
                    print "ch leven true"; print "ch t2 "$13; print "wh";
                }
            }
        }
    } END{print "q"}' "$PHASE_FILE" | sac > /dev/null 2>&1 || true
done
echo "Pick insertion complete."
