#!/bin/bash

echo "Install thirdparty repositories."
if [ ! -d "./scripts/thirdparty/commonroad_io/" ]; then
  cd ./scripts/thirdparty/

  # git commonroad_io
  git clone https://gitlab.lrz.de/tum-cps/commonroad_io.git

  # git commonroad-interactive-scenarios
  git clone https://gitlab.lrz.de/tum-cps/commonroad-interactive-scenarios.git

  # git interaction-dataset
  mkdir interaction-dataset
  cd ./interaction-dataset
  git clone https://github.com/interaction-dataset/interaction-dataset.git
  mv ./interaction-dataset ./interaction_dataset
  cd ../

  # git l5kit 
  # @note we install the source-code since pip install l5kit requires 
  #       numpy version <= 1.2.0, which may cause conflicts with other pkgs.
  git clone https://github.com/woven-planet/l5kit.git
  mv ./l5kit ./l5kit-devkit

  # kmeans-pytorch
  git clone https://github.com/subhadarship/kmeans_pytorch.git

  cd ../../
fi

echo "Install thirdparty dependencies."
# commonroad_io
pip install -r scripts/thirdparty/commonroad_io/requirements.txt

# others
pip install -r ./requirements.txt

