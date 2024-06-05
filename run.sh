conda activate dnerf
python run_nerf.py --config configs/chair.txt --finest_res 512 --log2_hashmap_size 19 --lrate 0.01 --lrate_decay 10 --testskip 8 --chunk 16384 --i_testset 1000