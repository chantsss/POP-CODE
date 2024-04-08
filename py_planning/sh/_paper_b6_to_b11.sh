#!/bin/bash

# B6-B8
./sh/predtest_conti+_k.sh
wait

# B9-B11
./sh/predtest_izone_lrc_irule_k.sh -0.01 1.0 3.0 -15.0
wait
