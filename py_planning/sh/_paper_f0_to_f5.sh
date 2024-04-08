#!/bin/bash

# F0-F5
./sh/paramtest_izone_lrc_irule.sh -15.0 0.5 0.01 -15.0
wait
./sh/paramtest_izone_lrc_irule.sh -15.0 0.5 0.5 -15.0
wait
./sh/paramtest_izone_lrc_irule.sh -15.0 0.5 1.0 -15.0
wait
./sh/paramtest_izone_lrc_irule.sh -15.0 0.5 2.0 -15.0
wait
./sh/paramtest_izone_lrc_irule.sh -15.0 0.5 3.0 -15.0
wait
./sh/paramtest_izone_lrc_irule.sh -15.0 0.5 5.0 -15.0
wait
