conda activate dnerf
python run_nerf.py --config configs/lego.txt --finest_res 512 --log2_hashmap_size 19 --lrate 0.01 --lrate_decay 10 --i_weights 500 --i_testset 500 --n_iters 1000

