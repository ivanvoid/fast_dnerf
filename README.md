# Fast D-NeRF
Dynamic NeRF but fast. (wip)

## Install

```
git clone https://github.com/ivanvoid/fast_dnerf
pip install -r requirements.txt
```


## Data
[Dynamic NeRF dataset](https://drive.google.com/file/d/1uHVyApwqugXTFuIRRlE4abTW8_rrVeIK/view)

## Run
Train model
```
conda deactivate
conda activate dnerf
. run.sh
```   
Run tensorboard:
```
~/miniconda3/envs/dnerf/bin/tensorboard --logdir=./logs 
```
Inference???

# Acknowledgement
This repo based on works of:

- [D-NeRF](https://github.com/albertpumarola/D-NeRF)
- [HashNeRF](https://github.com/yashbhalgat/HashNeRF-pytorch)


