#!/bin/bash

export model10_path='/home/abing/Workspace/experiment_ws/pyplanning/commonroad/models/net_pgp_gatx2_lvm_pp1.0_final'

if [ $# -eq 4 ];
then
  echo "Input with [constant mode value, ireact_gap_cond_t, ireact_delay_s, influ_react_min_acc] = [$1, $2, $3, $4]"
  # exp_izone
  python scripts/planning_commonroad.py --model_path ${model10_path} --is_exp_mode 2 --planner_type izone_rela --prediction_mode default --algo_variable1 $2 --algo_variable2 $3 --algo_variable3 $4 --acc_mode const --acc_const_value $1 --reaction_traj_mode irule --predict_traj_mode_num 3 --st_coefficents 1 --solu_tag_str coeftest_izone-lrcirule1C$1[$2,$3,$4]_1.0_p3;
  wait
  python scripts/planning_commonroad.py --model_path ${model10_path} --is_exp_mode 2 --planner_type izone_rela --prediction_mode default --algo_variable1 $2 --algo_variable2 $3 --algo_variable3 $4 --acc_mode const --acc_const_value $1 --reaction_traj_mode irule --predict_traj_mode_num 3 --st_coefficents 2 --solu_tag_str coeftest_izone-lrcirule2C$1[$2,$3,$4]_1.0_p3;
  wait
  python scripts/planning_commonroad.py --model_path ${model10_path} --is_exp_mode 2 --planner_type izone_rela --prediction_mode default --algo_variable1 $2 --algo_variable2 $3 --algo_variable3 $4 --acc_mode const --acc_const_value $1 --reaction_traj_mode irule --predict_traj_mode_num 3 --st_coefficents 3 --solu_tag_str coeftest_izone-lrcirule3C$1[$2,$3,$4]_1.0_p3;
  wait
  python scripts/planning_commonroad.py --model_path ${model10_path} --is_exp_mode 2 --planner_type izone_rela --prediction_mode default --algo_variable1 $2 --algo_variable2 $3 --algo_variable3 $4 --acc_mode const --acc_const_value $1 --reaction_traj_mode irule --predict_traj_mode_num 3 --st_coefficents 4 --solu_tag_str coeftest_izone-lrcirule4C$1[$2,$3,$4]_1.0_p3;
  wait

else
  echo "$0: Missing/Too much arguments"
  echo "list of variables= [$@]"
fi
