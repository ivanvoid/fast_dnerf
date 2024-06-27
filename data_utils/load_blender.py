import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2

from data_utils.utils import get_bbox3d_for_blenderobj

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    all_times = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        
        imgs = []
        poses = []
        times = []

        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for t, frame in enumerate(meta['frames'][::skip]):
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))

            # Time
            if 'time' in frame:
                cur_time = frame['time']
            else:
                cur_time = float(t) / (len(meta['frames'][::skip])-1)
            times.append(cur_time)

        assert times[0] == 0, "Time must start at 0"

        # keep all 4 channels (RGBA)
        imgs = (np.array(imgs) / 255.).astype(np.float32) 
        all_imgs.append(imgs)

        poses = np.array(poses).astype(np.float32)
        all_poses.append(poses)
        
        times = np.array(times).astype(np.float32)
        all_times.append(times)
        
        counts.append(counts[-1] + imgs.shape[0])

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    times = np.concatenate(all_times, 0)


    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    # Hash
    bounding_box = get_bbox3d_for_blenderobj(metas["train"], H, W, near=2.0, far=6.0)
        
    # Render times
    if os.path.exists(os.path.join(basedir, 'transforms_{}.json'.format('render'))):
        with open(os.path.join(basedir, 'transforms_{}.json'.format('render')), 'r') as fp:
            meta = json.load(fp)
        render_poses = []
        for frame in meta['frames']:
            render_poses.append(np.array(frame['transform_matrix']))
        render_poses = np.array(render_poses).astype(np.float32)
    else:
        render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    render_times = torch.linspace(0., 1., render_poses.shape[0])


    output_dict = {
        'images':imgs, 
        'poses':poses, 
        'render_poses':render_poses, 
        'hwf':[H, W, focal], 
        'i_split':i_split, 
        'bounding_box':bounding_box,
        'render_times':render_times,
        'times':times
    }

    return output_dict


class BlenderDataset:
    def __init__(self):
        pass
        # blender_data = load_blender_data(
        #         cfg.datadir, cfg.half_res, cfg.testskip)
        
        # images = blender_data['images']
        # poses = blender_data['poses']
        # render_poses = blender_data['render_poses']
        # hwf = blender_data['hwf']
        # i_split = blender_data['i_split']
        # bounding_box = blender_data['bounding_box']
        # times = blender_data['times']
        # render_times = blender_data['render_times']

        # cfg.bounding_box = bounding_box
        # print('Loaded blender', images.shape, render_poses.shape, hwf, cfg.datadir)
        # i_train, i_val, i_test = i_split

        # near = 2.
        # far = 6.

        # if cfg.white_bkgd:
        #     images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        # else:
        #     images = images[...,:3]

        # # Data time check
        # min_time, max_time = times[i_train[0]], times[i_train[-1]]
        # assert min_time == 0., "time must start at 0"
        # assert max_time == 1., "max time must be 1"

        # # Cast intrinsics to right types
        # H, W, focal = hwf
        # H, W = int(H), int(W)
        # hwf = [H, W, focal]

        # if K is None:
        #     K = np.array([
        #         [focal, 0, 0.5*W],
        #         [0, focal, 0.5*H],
        #         [0, 0, 1]
        #     ])

    def __getitem__(self, idx):
        pass
    def __len__(self):
        return 0