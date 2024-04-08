#!/bin/bash

# mode = 3
./sh/predtest_conti+.sh
wait
./sh/predtest_izone_lrc_irule2.sh -0.01 1.0 3.0 -15.0
wait
