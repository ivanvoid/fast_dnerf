# Fast-Dynamic-NeRF

![Dynamic NeRF Prediction](/_graphics/Predictions.gif)


## Setup
```
conda env create -f environment.yml
```

## Data
[Dynamic NeRF dataset](https://drive.google.com/file/d/1uHVyApwqugXTFuIRRlE4abTW8_rrVeIK/view)

Put data in `fast_dnerf/data/`

## Run
Train:
```
. train.sh
```

Inference:
```
. inference.sh
```

## Misc
To visualize cameras:
```
python camera_pose_visualizer.py
```

## Nodes
- Ray batching with timesteps

make a gif:
```
convert -delay 10 -loop 0 *.png ani.gif
```

## Acknowledgement
This repo based on works of:

- [D-NeRF](https://github.com/albertpumarola/D-NeRF)
- [HashNeRF](https://github.com/yashbhalgat/HashNeRF-pytorch)