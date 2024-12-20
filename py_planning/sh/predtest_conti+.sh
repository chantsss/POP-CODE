#!/bin/bash

export model005_path='/home/abing/Workspace/experiment_ws/pyplanning/commonroad/models/net_pgp_gatx2_lvm_pp0.05_final'
export model02_path='/home/abing/Workspace/experiment_ws/pyplanning/commonroad/models/net_pgp_gatx2_lvm_pp0.2_final'
export model04_path='/home/abing/Workspace/experiment_ws/pyplanning/commonroad/models/net_pgp_gatx2_lvm_pp0.4_final'
export model06_path='/home/abing/Workspace/experiment_ws/pyplanning/commonroad/models/net_pgp_gatx2_lvm_pp0.6_final'
export model08_path='/home/abing/Workspace/experiment_ws/pyplanning/commonroad/models/net_pgp_gatx2_lvm_pp0.8_final'
export model10_path='/home/abing/Workspace/experiment_ws/pyplanning/commonroad/models/net_pgp_gatx2_lvm_pp1.0_final'

python scripts/planning_commonroad.py --model_path ${model005_path} --is_exp_mode 2 --planner_type conti+ --prediction_mode default --predict_traj_mode_num 3 --solu_tag_str predtest_conti+_0.05_p3;
wait
python scripts/planning_commonroad.py --model_path ${model02_path} --is_exp_mode 2 --planner_type conti+ --prediction_mode default --predict_traj_mode_num 3 --solu_tag_str predtest_conti+_0.2_p3;
wait
python scripts/planning_commonroad.py --model_path ${model04_path} --is_exp_mode 2 --planner_type conti+ --prediction_mode default --predict_traj_mode_num 3 --solu_tag_str predtest_conti+_0.4_p3;
wait
python scripts/planning_commonroad.py --model_path ${model06_path} --is_exp_mode 2 --planner_type conti+ --prediction_mode default --predict_traj_mode_num 3 --solu_tag_str predtest_conti+_0.6_p3;
wait
python scripts/planning_commonroad.py --model_path ${model08_path} --is_exp_mode 2 --planner_type conti+ --prediction_mode default --predict_traj_mode_num 3 --solu_tag_str predtest_conti+_0.8_p3;
wait
# python scripts/planning_commonroad.py --model_path ${model10_path} --is_exp_mode 2 --planner_type conti+ --prediction_mode default --predict_traj_mode_num 3 --solu_tag_str predtest_conti+_1.0_p3;
# wait
