import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from hash_encoding import HashEmbedder, SHEncoder

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(args, emb_type='', input_dim=None, multires=None):
    '''
    emb_type str: Identity, SinCos, Hashm SHE
    '''
    assert emb_type != '', 'Select embedding function!'
    
    if emb_type == 'Identity':
        assert input_dim != None, 'Select input dimentions'
        return nn.Identity(), input_dim
    
    elif emb_type == 'SinCos':
        assert multires != None, 'Select Multiresolution'
        assert input_dim != None, 'Select input dimentions'
        embed_kwargs = {
                    'include_input' : True,
                    'input_dims' : input_dim,
                    'max_freq_log2' : multires-1,
                    'num_freqs' : multires,
                    'log_sampling' : True,
                    'periodic_fns' : [torch.sin, torch.cos],
        }    
        embedder_obj = Embedder(**embed_kwargs)
        embed = lambda x, eo=embedder_obj : eo.embed(x)
        out_dim = embedder_obj.out_dim

    elif emb_type == 'Hash':
        embed = HashEmbedder(bounding_box=args.bounding_box, \
                            log2_hashmap_size=args.log2_hashmap_size, \
                            finest_resolution=args.finest_res)
        out_dim = embed.out_dim
        
    elif emb_type == 'SHE':
        embed = SHEncoder()
        out_dim = embed.out_dim
    
    return embed, out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))


# Small NeRF for Hash embeddings
class NeRFSmall(nn.Module):
    def __init__(self,
                 num_layers=3,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=4,
                 hidden_dim_color=64,
                 input_ch=3, input_ch_views=3,
                 ):
        super(NeRFSmall, self).__init__()

        self.input_ch = input_ch
        self.input_ch_views = input_ch_views

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.input_ch
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim
            
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        
        color_net =  []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.input_ch_views + self.geo_feat_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = hidden_dim
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)
    
    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        # sigma
        h = input_pts
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        sigma, geo_feat = h[..., 0], h[..., 1:]
        
        # color
        h = torch.cat([input_views, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
            
        # color = torch.sigmoid(h)
        color = h
        outputs = torch.cat([color, sigma.unsqueeze(dim=-1)], -1)

        return outputs


### Hash tempootal NERF
class DirectTemporalNeRFSmall(nn.Module):
    def __init__(
            self,
            n_layers=3,
            hidden_dim=64,
            geo_feat_dim=15,
            n_layers_color=4,
            hidden_dim_color=64,
            input_dim=3, 
            input_dim_views=3,
            input_dim_time=3,
            ):
        super(DirectTemporalNeRFSmall, self).__init__()

        self.input_dim = input_dim
        self.input_dim_views = input_dim_views

        ###
        # Transperency net (Sigma network)
        self.num_layers = n_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        sigma_net = []
        for l in range(n_layers):
            if l == 0:
                in_dim = self.input_dim
            else:
                in_dim = hidden_dim
            if l == n_layers - 1:
                # 1 sigma + 15 SH features for color
                out_dim = 1 + self.geo_feat_dim 
            else:
                out_dim = hidden_dim
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))
        self.sigma_net = nn.ModuleList(sigma_net)

        ###
        # Color network
        self.num_layers_color = n_layers_color        
        self.hidden_dim_color = hidden_dim_color
        
        color_net =  []
        for l in range(n_layers_color):
            if l == 0:
                in_dim = self.input_dim_views + self.geo_feat_dim
            else:
                in_dim = hidden_dim
            if l == n_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = hidden_dim
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))
        self.color_net = nn.ModuleList(color_net)

        ###
        # Temporal network
        self.input_dim_time = input_dim_time
        self.time_net_skip = []
        self.time_net = self.create_time_net()
    
    def create_time_net(self):
        layers = [nn.Linear(
            self.input_dim + self.input_dim_time, 
            self.hidden_dim)]
        
        for i in range(self.num_layers - 1):
            in_channels = self.hidden_dim
            if i in self.time_net_skip:
                in_channels += self.input_dim

            layers += [nn.Linear(in_channels, self.hidden_dim)]
        # Dim 3 - Coordinates of points in 3D space
        # layers += [nn.Linear(self.hidden_dim, 3)]
        layers += [nn.Linear(self.hidden_dim, self.input_dim)]    
        return nn.ModuleList(layers)

    def forward_time_net(self, points, timestep):
        assert len(torch.unique(timestep[:, :1])) == 1, "Only accepts all points from same time"

        # import pdb;pdb.set_trace()
        x = torch.cat([points, timestep], dim=-1)

        for i, layer in enumerate(self.time_net):
            x = layer(x)
            x = F.relu(x)
            if i in self.time_net_skip:
                x = torch.cat([points, x], -1)
        return x


    def forward(self, x):
        # import pdb;pdb.set_trace()
        points, views, timestep = torch.split(
            x, 
            [self.input_dim, self.input_dim_views, self.input_dim_time], 
            dim=-1)

        # Forward time 
        cur_time = timestep[0, 0]
        if cur_time == 0:
            # Original points
            dx = torch.zeros_like(points[:, :3])
        else:
            dx = self.forward_time_net(points, timestep)
            # original_points = points[:, :3]
            # points = self.embed_fn(original_points + dx)
            points = points + dx

        # sigma
        h = points
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        sigma, geo_feat = h[..., 0], h[..., 1:]
        
        # color
        h = torch.cat([views, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
            
        # color = torch.sigmoid(h)
        color = h
        outputs = torch.cat([color, sigma.unsqueeze(dim=-1)], -1)

        return outputs


# n_layers=2,
# hidden_dim=64,
# geo_feat_dim=15,
# n_layers_color=3,
# hidden_dim_color=64,
# input_dim=embeddings['point_dim'], 
# input_dim_views=embeddings['views_dim'],
# input_dim_time=embeddings['time_dim'])

class FD_NeRF(nn.Module):
    def __init__(self,
                 n_layers=3,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 n_layers_color=4,
                 hidden_dim_color=64,
                 input_dim_points=3, 
                 input_dim_views=3,
                 input_dim_time=1,
                 output_dim=4
                 ):
        """ 
        """
        super(FD_NeRF, self).__init__()
        
        ### Set NeRF Network
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.input_dim_points = input_dim_points
        self.input_dim_views = input_dim_views
        self.input_dim_time = input_dim_time

        self.geo_feat_dim = geo_feat_dim
        self.n_layers_color = n_layers_color
        self.hidden_dim_color = hidden_dim_color

        self.skips = []
        
        linears = [nn.Linear(input_dim_points, self.hidden_dim)]
        for i in range(self.n_layers):
            if i not in self.skips:
                l = nn.Linear(self.hidden_dim, self.hidden_dim)
            else:
                l = nn.Linear(self.hidden_dim + input_dim_points, self.hidden_dim)
            linears += [l]
        self.linears = nn.ModuleList(linears)
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([
            nn.Linear(input_dim_views + self.hidden_dim, self.hidden_dim//2)])
        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        use_viewdirs = True
        self.use_viewdirs = use_viewdirs
        if use_viewdirs:
            self.feature_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.alpha_linear = nn.Linear(self.hidden_dim, 1)
            self.rgb_linear = nn.Linear(self.hidden_dim//2, 3)
        else:
            self.output_linear = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x):
        points, views, timestep = torch.split(
            x, 
            [self.input_dim_points, self.input_dim_views, self.input_dim_time], 
            dim=-1)
        # input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = points
        for i, l in enumerate(self.linears):
            h = self.linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([points, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs   