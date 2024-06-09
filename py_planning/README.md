# IPM-Planning/

## 1 Installation

The following commands were checked in Python3.8.

- Create a conda environment and activate it.

  ```
  conda create -n your_conda_env python=3.8
  conda activate your_conda_env
  ```

  

- Install repositories & dependencies.

  ```
  ./install.sh
  ```

  The following repositories & their dependencies will be installed:

  ```
  1. commonroad
  2. commonroad-interactive-scenarios
  3. NuScenese
  ```

  @note if it is your first time to install **sumo**, it is required to reboot the computer. For details, please see README.md in **thirdparty/commonroad-interactive-scenarios**.

  

- manually install pytorch in your env, for example:

  Corresponding to **nvcc --version**: release 11.3
  
  ```
  conda install pytorch torchvision torchaudio tensorboard cudatoolkit=11.3 -c pytorch
  ```
  
  @note cudatoolkit version (11.3 or else) should be in line with your hardware requirements.
  



- manually install other required pkgs

  ```
  sudo apt install sumo
  ```

  

## 2 Preparation

- **vscode setting.json**: which is used for code completio in **vscode**.

  ```
  "your/path/to/this_repository/thirdparty/commonroad_io/",
  "your/path/to/this_repository/thirdparty/nuscenes-devkit/python-sdk/",
  "your/path/to/this_repository/thirdparty/commonroad-interactive-scenarios/",
  "your/path/to/this_repository/thirdparty/interaction-dataset/interaction_dataset/",
  "your/path/to/this_repository/thirdparty/l5kit-devkit/l5kit/",
  "your/path/to/this_repository/thirdparty/kmeans_pytorch/",
  ```
  
  
  
- To invoke functions from **third parties** in .py, import the thirdparty.config first. 

  For example:
  
  ```
  import thirdparty.config
  
  # and here can use resnetbackbond from nuscenes lib
  from nuscenes.prediction.models.backbone import ResNetBackbone
  ```
  
  
  
- Add the following **environment variables** (in ~/.bashrc):

  @note some environment variables are not compulsory, it depends on which function is applied.

  ```
  export EXP_ROOT="/path/to/experiment_folder";
  
  export COMMONROAD_DATA_ROOT="/path/to/commonroad/commonroad-scenarios";
  export NUSCENES_DATA_ROOT="/path/to/nuScenes";
  export INTERACTION_DATA_PREDICTION_ROOT="/path/to/INTERACTION_DATASET_PREDICTION_PART";
  export L5KIT_DATA_ROOT="/path/to/l5kit"
  
  export SUMO_HOME=/usr/bin/sumo # your/path/to/sumo
  ```
  
  1. **EXP_ROOT**
  
     An arbitrary path to store experiment results.
  
     
  
  2. **NUSCENES_DATA_ROOT (not supported yet)** 
  
     The path to store the nuscenes dataset (**trajectory prediction** & **trajectory set**) from 
  
     https://www.nuscenes.org/nuscenes.
  
     The file structure looks like:
  
     ```
     nuScenes/
     ├── v1.0-mini/
     │   ├── maps/
     │   ├── samples/
     │   ├── sweeps/
     │   └── v1.0-mini/
     └── v1.0-trainval/
     │   ├── maps/
     │   └── v1.0-trainval/
     └── nuscenes-prediction-challenge-trajectory-sets
         ├── epsilon_2.pkl
         ├── epsilon_4.pkl
         └── epsilon_8.pkl
     ```
  
     
  
  3. **COMMONROAD_DATA_ROOT**
  
     The path to store the commonroad-scenarios/
  
     https://gitlab.lrz.de/tum-cps/commonroad-scenarios.
  
     The file structure looks like:
  
     ```
     commonroad/
     └── commonroad-scenarios/
     ```
     
     
     
  4. **INTERACTION_DATA_PREDICTION_ROOT**
  
     The path to store INTERACTION DATASET prediction.
  
     https://interaction-dataset.com/.
  
     The file structure looks like:
  
     ```
     INTERACTION-Dataset-Prediction/
     ├── INTERACTION-Dataset-DR-multi-v1_2
     │   ├── maps
     │   ├── test_conditional-multi-agent
     │   ├── test_multi-agent
     │   ├── train
     │   └── val
     └── INTERACTION-Dataset-DR-single-v1_2
     ```
  
     
  
  3. **L5KIT_DATA_ROOT (not supported yet)** 
  
     The path to store l5kit dataset.
  
     https://woven-planet.github.io/l5kit/.
  
     The file structure looks like:
  
     ```
     l5kit/
     ├── aerial_map/
     ├── scenes
     ├── semantic_map
     └── meta.json
     ```



## 3 How to use

First step is to activate the conda env

```
conda activate your_conda_env
```



### a. Test of third-party interafces

Normal test, test the availabilities of third-parties. 

```
python3 scripts/tests/test_thirdparty_interfaces.py
```



### b. Generate scenarios from commonroad-sumo simulation

This step will generate scenario (-T) files in **EXP_ROOT/commonroad/simulated_scenarios** folder by using commonroad-sumo interface to simulate the (-I) scenarios.



Run

```
python scripts/extract_commonroad_scenarios_from_simu.py
```

- TIP1: this step is not compulsory. However, the scenario data offered by commonroad in COMMONROAD_DATA_ROOT/scenarios/recorded/xxx/ is not enough to train the prediction models, since "agents" in closed-loop sumo simulation use a different strategy compared to those in recorded scenarios.
- TIP2: In commonroad, scenario with (-T) is with true values, (-I) is for simulation, (-S) is with set-based prediction, and so on. For more details, please check https://commonroad.in.tum.de.



To configure where to read scenario.xml files for simulation, please see:

```
envs.commonroad.simulation.config.py

	SCENE_DIRS = []
```



### c. Extract data for training prediction network

This step is to extract prediction train/eval torch.tensor files from commonroad scenario.xml files for training the prediction network.  

Run the following command:

```
python scripts/process_datasets.py --dataset commonroad --process prediction --num_workers 10
```

It will generate data for training prediction module, and the results will be stored at **EXP_ROOT/commonroad/prediction/** folder.


The following **yaml** file is used in configuring the vectorization & graph:

```
envs/conf/preprocess_dataset_config.yaml
```

And the following **json** files are used in configuring the raster image:

```
envs/commonroad/conf/focused_dyn_obst_params.json # configs for target agent
envs/commonroad/conf/raster_params.json # configs for other dynamic agents
```



#### Extract more data

Likewise, use the following code to extract trajectories from the commonroad scenarios:

```
python scripts/process_datasets.py --dataset commonroad --process trajectory --num_workers 1
```

And the data will be stored at **EXP_ROOT/commonroad/trajectory/** folder.



### d. Analyze data and visualization

To analyze and visualize the extracted data in **3. Extract data for training prediction network **

Run the following command to visualize trajectories

```
python scripts/analyze_processed_dataset.py --datasets 'commonroad' --read_mode trajectory
```

And interaction data, as

```
python scripts/analyze_processed_dataset.py --datasets 'interaction_dataset' --read_mode interaction
```

### e. Train & Evaluate prediction models

#### 1) Training

After complete **3. Extract data for training prediction network**, run the following commands to train a prediction model. 

> @note we have tested mtp and covernet networks from **Nuscenes devkit**, and the prediction result is not good as pgp encoder based networks.



The following command support the network training

```
python scripts/train_prediction_model.py -c ./scripts/envs/conf/net_pgp_gatx2_lvm_traversal.yml -m pgp_lvm_trav -n 200
```

where, the network structure is configured at **net_pgp_gatx2_lvm_traversal.yml**, and the "-m" denotes the model folder name being saved. 



For example, the dataset is configured at net_pgp_gatx2_lvm_traversal.yml and the trained model is saved at **EXP_ROOT/${dataset}/models/**. 

More configurations are loaded at 

```
scripts/envs/conf/prediction_train_config.yaml:

commonroad: 
  # Which mode to excute
  #   1. extract: extract data from dataset ('data_path').
  #   2. load: try load data from 'save_path'.
  # @note data_path and save_path are defined in process_dataset.py
  mode: 'load'
  version: 'train'
  # Percentage of data being loaded in the train
  full_loaded: 0.05
```



For a trained model, configuration file and the checkpoint files are stored

```
./
├── checkpoints
├── config.yaml
└── tensorboard_logs
```



#### 2) Evaluating and comparing trained models

Run the following command,

```
python scripts/eval_prediction_models.py -m /your/EXP_ROOT/commonroad/models/
```

And all trained prediction models will be evaluated and compared, example result like

```
[100.0%]: [8]min_ade_5: 0.79, [8]min_ade_10: 0.59, [12]min_ade_10: 1.17, [8]miss_rate_5: 0.26, [8]miss_rate_10: 0.15, [12]miss_rate_10: 0.44, pi_bc: 2.52, ;
```



where, the [percentage %] means how much percentage of data are used in trainning.






### f. Closed-loop simulation with planner

```
python scripts/planning_commonroad.py --model_path /your/EXP_ROOT/$dataset$/models/net_pgp_gatx2_lvm_pp1.0_final --check_point best --create_video 1

```

- --model_path: the path to the prediction model being applied during closed-loop simulation

- --check_point: the check point tag of the prediction model

- --create_video: int = 1 (enable), = 0 (disable); enable create pictures in **EXP_ROOT/${dataset}/exp_plan/videos/${tag_str}** or not, 

  where "tag_str" is defined as **solu_tag_str** in scripts/conf/planning_commonroad.yaml.
  
  **Note**: create_video == 1, the latex supporting environment is supported in your environment. To disable this, comment the following code in *scripts/paper_plot/utils.py*:
  
  ```
  modify 
  	plt.rc('text', usetex=True) 
  as
  	plt.rc('text', usetex=False)
  or
  	# plt.rc('text', usetex=True) 
  ```



During simulations, files of result scenarios and created in **EXP_ROOT/${dataset}/exp_plan/**, including

```
EXP_ROOT/${dataset}/exp_plan/result_scenarios/${tag_str}

EXP_ROOT/${dataset}/exp_plan/solutions/${tag_str}

EXP_ROOT/${dataset}/exp_plan/videos/${tag_str}
```

In addition,  prediction module in the simulation uses the configurations in **envs/conf/preprocess_simu_config.yaml**. 

Here is an example to create GIFs from images **create_video.py**


### g. Evaluate closed-loop simulation performance

After finish simulation, run

```
python scripts/eval_commonroad_solutions.py
```

Which will evaluate solutions with [tag_str, ...] in scripts/conf/eval_tags.yaml, e.g., 

And the performance metrics and records will be stored at **EXP_ROOT/${dataset}/exp_plan/evals/${tag_str}$** for further visualization.



## 4 Other Useful Commands

- kill all ray processes when it is needed

  @note the code do not use the ray lib anymore

  ```
  - ps aux | grep ray::IDLE | grep -v grep | awk '{print $2}' | xargs kill -9
  ```


