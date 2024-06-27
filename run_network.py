import os
import torch
import time
import tqdm
import numpy as np
import matplotlib.pyplot as plt

from render import *
from run_nerf_helpers import to8b
###
### Run network
###

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

def stupid_bachify(model, chunk, inputs):
    result = []
    for i in range(0, inputs.shape[0], chunk):
        prediction = model(inputs[i:i+chunk])
        result += [prediction]
    result = torch.cat(result, 0)
    return result

def run_network(inputs, viewdirs, timestep, model, embed_fn, embeddirs_fn, embedtime_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    # Position embedding
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded_points, keep_mask = embed_fn(inputs_flat)
    
    # Views embedding
    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)

    # Time embedding
    assert len(torch.unique(timestep)) == 1, "Only accepts all points from same time"
    B, N, _ = inputs.shape
    input_timestep = timestep[:, None].expand([B, N, 1])
    input_timestep_flat = torch.reshape(input_timestep, [-1, 1])
    embedded_time = embedtime_fn(input_timestep_flat)

    # Combine embeddings
    embedded = torch.cat([embedded_points, embedded_dirs, embedded_time], -1)

    # Forward pass
    # outputs_flat = batchify(fn, netchunk)(embedded)
    outputs_flat = stupid_bachify(model, netchunk, embedded)
    outputs_flat[~keep_mask, -1] = 0 # set sigma to 0 for invalid points
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf
    near, far = render_kwargs['near'], render_kwargs['far']

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    depths = []
    psnrs = []
    accs = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        # print(i, time.time() - t)
        t = time.time()
        rgb, depth, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        # normalize depth to [0,1]
        depth = (depth - near) / (far - near)
        depths.append(depth.cpu().numpy())

        accs.append(acc.cpu().numpy())
        # if i==0:
        #     print(rgb.shape, depth.shape)

        if gt_imgs is not None and render_factor==0:
            try:
                gt_img = gt_imgs[i].cpu().numpy()
            except:
                gt_img = gt_imgs[i]
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_img)))
            # print(p)
            psnrs.append(p)

        if savedir is not None:
            # save rgb and depth as a figure
            fig = plt.figure(figsize=(25,15))
            ax = fig.add_subplot(1, 2, 1)
            rgb8 = to8b(rgbs[-1])
            ax.imshow(rgb8)
            ax.axis('off')
            ax = fig.add_subplot(1, 2, 2)
            ax.imshow(depths[-1], cmap='plasma', vmin=0, vmax=1)
            ax.axis('off')
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            # save as png
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            # imageio.imwrite(filename, rgb8)
    
    rgbs = np.array(rgbs)
    depths = np.array(depths)
    psnrs = np.array(psnrs)
    accs = np.array(accs)
    return rgbs,depths,psnrs,accs


    rgbs = np.stack(rgbs, 0)
    depths = np.stack(depths, 0)
    if gt_imgs is not None and render_factor==0:
        avg_psnr = sum(psnrs)/len(psnrs)
        print("Avg PSNR over Test set: ", avg_psnr)
        with open(os.path.join(savedir, "test_psnrs_avg{:0.2f}.pkl".format(avg_psnr)), "wb") as fp:
            pickle.dump(psnrs, fp)

    return rgbs, depths
