#!/bin/bash

export model10_path='/home/abing/Workspace/experiment_ws/pyplanning/commonroad/models/net_pgp_gatx2_lvm_pp1.0_final'

python scripts/planning_commonroad.py --model_path ${model10_path} --is_exp_mode 2 --planner_type conti+ --prediction_mode default --predict_traj_mode_num 3 --st_coefficents 1 --solu_tag_str coeftest_conti+1C_1.0_p3;
wait
python scripts/planning_commonroad.py --model_path ${model10_path} --is_exp_mode 2 --planner_type conti+ --prediction_mode default --predict_traj_mode_num 3 --st_coefficents 2 --solu_tag_str coeftest_conti+2C_1.0_p3;
wait
python scripts/planning_commonroad.py --model_path ${model10_path} --is_exp_mode 2 --planner_type conti+ --prediction_mode default --predict_traj_mode_num 3 --st_coefficents 3 --solu_tag_str coeftest_conti+3C_1.0_p3;
wait
python scripts/planning_commonroad.py --model_path ${model10_path} --is_exp_mode 2 --planner_type conti+ --prediction_mode default --predict_traj_mode_num 3 --st_coefficents 4 --solu_tag_str coeftest_conti+4C_1.0_p3;
wait

