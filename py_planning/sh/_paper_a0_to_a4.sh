#!/bin/bash

# A0
./sh/run_ca+.sh 
wait

# A1
./sh/run_ca.sh 
wait

# A2, A3
./sh/run_izone_lrc_pred_test.sh -15.0
wait

# A4 + A6
./sh/run_izone_lrc_pred.sh -15.0
wait
