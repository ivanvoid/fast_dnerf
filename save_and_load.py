import os
import torch
###
### Save and Load
###

def load_checkpoints(args, coarse_model, fine_model, embeddings, optimizer):
    start = 0
    basedir = args.basedir
    expname = args.expname

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        coarse_model.load_state_dict(ckpt['network_fn_state_dict'])
        if fine_model is not None:
            fine_model.load_state_dict(ckpt['network_fine_state_dict'])
        if args.i_embed==1:
            embeddings['point_fn'].load_state_dict(ckpt['embed_fn_state_dict'])
    
    return start
