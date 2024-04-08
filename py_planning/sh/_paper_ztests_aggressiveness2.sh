#!/bin/bash

# mode = 3
./sh/coeftest_conti+.sh
wait
./sh/coeftest_izone_lrc_irule2.sh -0.01 1.0 3.0 -15.0
wait
