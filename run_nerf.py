import os, sys

import numpy as np
import imageio
import json
import pdb
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm, trange
import pickle

from torch.utils.tensorboard import SummaryWriter

### My files
# DATA
from data_utils.load_llff import load_llff_data
from data_utils.load_deepvoxels import load_dv_data
from data_utils.load_blender import load_blender_data
from data_utils.load_scannet import load_scannet_data
from data_utils.load_LINEMOD import load_LINEMOD_data

# 
from optimizer import MultiOptimizer

from loss import sigma_sparsity_loss, total_variation_loss
from run_nerf_helpers import *

#
from render import *
from arg_parser_to_config import get_config
from create_objects import *
from run_network import *
from save_and_load import *
from logging_setter import set_logging


DEBUG = False


def inference(args):
    print('RENDER ONLY')
    # with torch.no_grad():
    #     if args.render_test:
    #         # render_test switches to test poses
    #         images = images[i_test]
    #     else:
    #         # Default is smoother render_poses path
    #         images = None

    #     testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
    #     os.makedirs(testsavedir, exist_ok=True)
    #     print('test poses shape', render_poses.shape)

    #     rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
    #     print('Done rendering', testsavedir)
    #     imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

    #     return
    pass


def load_data(args):
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test, bounding_box = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        args.bounding_box = bounding_box
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)

        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.

        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        blender_data = load_blender_data(
            args.datadir, args.half_res, args.testskip)
        
        images = blender_data['images']
        poses = blender_data['poses']
        render_poses = blender_data['render_poses']
        hwf = blender_data['hwf']
        i_split = blender_data['i_split']
        bounding_box = blender_data['bounding_box']
        times = blender_data['times']
        render_times = blender_data['render_times']

        args.bounding_box = bounding_box
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'scannet':
        images, poses, render_poses, hwf, i_split, bounding_box = load_scannet_data(args.datadir, args.scannet_sceneID, args.half_res)
        args.bounding_box = bounding_box
        print('Loaded scannet', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 0.1
        far = 10.0

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return
    
    # Data time check
    min_time, max_time = times[i_train[0]], times[i_train[-1]]
    assert min_time == 0., "time must start at 0"
    assert max_time == 1., "max time must be 1"

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])



    dataset = {
        'near':near,
        'far':far,
        'H':H,
        'W':W,
        'hwf':hwf,
        'K':K, # intrinsic matrix
        # extrinsic matrix
        'i_train':i_train,
        'i_val':i_val, 
        'i_test':i_test,
        'render_poses':render_poses,
        'render_times':render_times,
        'poses':poses,
        'times':times,
        'images':images,
    }

    return dataset


###
###
###
def train():
    # DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CONFIG
    args = get_config()

    # LOAD DATASET
    ds = load_data(args)

    # LOGGING SETUP
    basedir, expname = set_logging(args)

    # CREATE NERF MODEL
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args, device)
    global_step = start
    

    # ???
    bds_dict = {
        'near' : ds['near'],
        'far' :  ds['far'],
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(ds['render_poses']).to(device)
    render_times = torch.Tensor(ds['render_times']).to(device)

    
    # Short circuit if only rendering out from trained model
    if args.render_only:
        inference()
        raise NotImplementedError
    

    # Prepare raybatch tensor if batching random rays
    # print('BATCHING: ',args.no_batching)
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        raise NotImplementedError
        # For random ray batching
        # print('get rays')
        # rays = np.stack([get_rays_np(ds['H'], ds['W'], ds['K'], p) for p in ds['poses'][:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        # print('done, concats')
        # rays_rgb = np.concatenate([rays, ds['images'][:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        # rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        # rays_rgb = np.stack([rays_rgb[i] for i in ds['i_train']], 0) # train images only
        # rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        # rays_rgb = rays_rgb.astype(np.float32)
        # print('shuffle rays')
        # np.random.shuffle(rays_rgb)

        # print('done')
        # i_batch = 0

    # Move training data to GPU
    if use_batching:
        ds['images'] = torch.Tensor(ds['images']).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)
    poses = torch.Tensor(ds['poses']).to(device)
    times = torch.Tensor(ds['times']).to(device)

    N_iters = 50000 + 1
    print('Begin')
    print('TRAIN views are', ds['i_train'])
    print('TEST views are', ds['i_test'])
    print('VAL views are', ds['i_val'])

    # Summary writers
    writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

    loss_list = []
    psnr_list = []
    time_list = []
    start = start + 1
    time0 = time.time()

    for i in trange(start, N_iters):
        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(ds['i_train'])
            target = ds['images'][img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3,:4]
            timestep = times[img_i]

            if N_rand is not None:
                # (H, W, 3), (H, W, 3)
                rays_o, rays_d = get_rays(ds['H'], ds['W'], ds['K'], torch.Tensor(pose))  

                if i < args.precrop_iters:
                    dH = int(ds['H']//2 * args.precrop_frac)
                    dW = int(ds['W']//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(ds['H']//2 - dH, ds['H']//2 + dH - 1, 2*dH),
                            torch.linspace(ds['W']//2 - dW, ds['W']//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")
                else:
                    coords = torch.stack(torch.meshgrid(
                        torch.linspace(0, ds['H']-1, ds['H']), 
                        torch.linspace(0, ds['W']-1, ds['W'])), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        
        rgb, depth, acc, extras = render(
            ds['H'], ds['W'], ds['K'], 
            timestep=timestep,
            chunk=args.chunk, 
            rays=batch_rays,
            verbose=i < 10, 
            retraw=True,
            **render_kwargs_train)

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][...,-1]
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        sparsity_loss = args.sparse_loss_weight*(extras["sparsity_loss"].sum() + extras["sparsity_loss0"].sum())
        loss = loss + sparsity_loss

        # add Total Variation loss
        if args.i_embed==1:
            n_levels = render_kwargs_train["embed_fn"].n_levels
            min_res = render_kwargs_train["embed_fn"].base_resolution
            max_res = render_kwargs_train["embed_fn"].finest_resolution
            log2_hashmap_size = render_kwargs_train["embed_fn"].log2_hashmap_size
            TV_loss = sum(total_variation_loss(render_kwargs_train["embed_fn"].embeddings[i], \
                                              min_res, max_res, \
                                              i, log2_hashmap_size, \
                                              n_levels=n_levels) for i in range(n_levels))
            loss = loss + args.tv_loss_weight * TV_loss
            if i>1000:
                args.tv_loss_weight = 0.0

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        t = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            if args.i_embed==1:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'embed_fn_state_dict': render_kwargs_train['embed_fn'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            else:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            print('Saved checkpoints at', path)

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, ds['hwf'], ds['K'], args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[ds['i_test']].shape)
            with torch.no_grad():
                rgbs, depths, psnrs, accs = render_path(
                    torch.Tensor(poses[ds['i_test']]).to(device), 
                    ds['hwf'], 
                    ds['K'], 
                    args.chunk, 
                    render_kwargs_test, 
                    gt_imgs=ds['images'][ds['i_test']], 
                    savedir=testsavedir)
            
            writer.add_image('gt', to8b(ds['images'][ds['i_test']][0]), i, dataformats='HWC')
            writer.add_image('rgb', to8b(rgbs[0]), i, dataformats='HWC')
            writer.add_image('depth', depths[0], i, dataformats='HW')
            writer.add_image('acc', accs[0], i, dataformats='HW')
            writer.add_scalar('psnr_test', psnrs[0], i)
            print('Saved test set')



        if i%args.i_print==0:
            writer.add_scalar('Loss', loss.item(), i)
            writer.add_scalar('PSNR', psnr.item(), i)
            # writer.add_scalar('loss', t, i)
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
            loss_list.append(loss.item())
            psnr_list.append(psnr.item())
            time_list.append(t)
            loss_psnr_time = {
                "losses": loss_list,
                "psnr": psnr_list,
                "time": time_list
            }
            with open(os.path.join(basedir, expname, "loss_vs_time.pkl"), "wb") as fp:
                pickle.dump(loss_psnr_time, fp)
        
        writer.flush()
        global_step += 1


def main():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.manual_seed(2024)
    np.random.seed(2024)

    train()

    return

if __name__=='__main__':
    main()
