#!/bin/bash

# mode = 1
./sh/predtest_izone_lrc_pred_nn.sh -15.0
wait
./sh/predtest_izone_lrc_irule.sh -0.01 1.0 3.0 -15.0
wait
