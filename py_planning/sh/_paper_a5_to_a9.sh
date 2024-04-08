#!/bin/bash

# A5 + A9
./sh/run_izone_lrc_irule.sh -0.01 1.0 3.0 -15.0
wait

# A6: fun by A4's command

# A7
./sh/run_izone_lrc_pred_lonshort.sh -15.0
wait

# A8
./sh/run_conti+.sh
wait

# A9: run by A5's command
