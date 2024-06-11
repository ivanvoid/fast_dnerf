import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

from hash_encoding import HashEmbedder, SHEncoder


class TemporalHashDnerf(nn.Module):
    def __init__(
            self,
            n_layers_deform=3,
            n_layers_sigma=3,
            n_layers_color=4,
            
            input_dim_points=3, 
            input_dim_views=3,
            input_dim_time=3,
            
            hidden_dim_deform=64,
            hidden_dim_sigma=64,
            hidden_dim_color=64,
            geo_feat_dim=15,
            
            encoding_1=None,
            encoding_1_dim=None,
            encoding_2=None,
            encoding_2_dim=None,
            encoding_3=None,
            encoding_3_dim=None,
            ):
        super(TemporalHashDnerf, self).__init__()

        # Input dimentions
        self.input_dim_points = input_dim_points
        self.input_dim_views = input_dim_views
        self.input_dim_time = input_dim_time

        # Number of layers for each net
        self.n_layers_deform = n_layers_deform
        self.n_layers_sigma = n_layers_sigma
        self.n_layers_color = n_layers_color

        # Hidden dimentions for each net
        self.hidden_dim_deform = hidden_dim_deform
        self.hidden_dim_sigma = hidden_dim_sigma
        self.hidden_dim_color = hidden_dim_color
        self.geo_feat_dim = geo_feat_dim
    
        # Encoding calls
        self.encoding_lambda_call=encoding_lambda_call
        self.encoding_lambda_dim = encoding_lambda_dim

        # Transperency net (Sigma network)
        self.sigma_net = self._init_sigma_net()

        # Color network
        self.color_net = self._init_color_net()

        # Temporal network
        self.time_net = self._init_time_net()
    
    def _init_sigma_net(self):
        in_dim = self.input_dim_points + self.input_dim_time + self.encoding_lambda_dim

        sigma_net = []
        for l in range(self.n_layers_sigma):
            if l != 0:
                in_dim = self.hidden_dim_sigma
            if l == self.n_layers_sigma - 1:
                # 1 sigma + 15 SH features for color
                out_dim = 1 + self.geo_feat_dim 
            else:
                out_dim = self.hidden_dim_sigma
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))
        return nn.ModuleList(sigma_net)

    def _init_color_net(self):
        in_dim = self.input_dim_views + self.geo_feat_dim

        color_net =  []
        for l in range(self.n_layers_color):
            if l != 0:
                in_dim = self.hidden_dim_color
            if l == self.n_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = self.hidden_dim_color
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))
        return nn.ModuleList(color_net)

    def _init_time_net(self): 
        in_dim = self.input_dim_points + self.input_dim_time

        time_net = []
        for l in range(self.n_layers_deform):
            if l != 0:
                in_dim = self.hidden_dim_deform
            if l == self.n_layers_deform - 1:
                out_dim = 3
            else:
                out_dim = self.hidden_dim_deform
            time_net.append(nn.Linear(in_dim, out_dim))
        return nn.ModuleList(time_net)

    def forward_time_net(self, deform):
        for i, layer in enumerate(self.time_net):
            deform = layer(deform)
            if i != self.n_layers_deform - 1:
                deform = F.relu(deform)
        return deform

    def forward_sigma_net(self, h):
        for l in range(self.n_layers_sigma):
            h = self.sigma_net[l](h)
            if l != self.n_layers_sigma - 1:
                h = F.relu(h, inplace=True)

        sigma, geo_feat = h[..., 0], h[..., 1:]
        return sigma, geo_feat

    def forward_color_net(self, h):
        for l in range(self.n_layers_color):
            h = self.color_net[l](h)
            if l != self.n_layers_color - 1:
                h = F.relu(h, inplace=True)
        color = torch.sigmoid(h)
        return color


    def forward(self, x):
        '''
        x:
            points - encoded    (N,32)
            views - encoded     (N,16)
            timestep - encoded  (N,21)
        '''
        points_emb, views_emb, timestep_emb = torch.split(
            x, 
            [self.input_dim_points, self.input_dim_views, self.input_dim_time], 
            dim=-1)
        
        # Time deformation
        assert len(torch.unique(timestep_emb[:, :1])) == 1, "Only accepts all points from same time"
        points_timestep = torch.cat([points_emb, timestep_emb], axis=-1)
        deform = self.forward_time_net(points_timestep)

        original_points = points_emb[:, :3]
        deformed_points = original_points + deform
        deformed_points_emb = self.encoding_lambda_call(deformed_points)

        h = torch.cat([points_emb, deformed_points_emb, timestep_emb], axis=-1)
        
        # Sigma
        sigma, geo_feat = self.forward_sigma_net(h)
        
        # Color
        h = torch.cat([views_emb, geo_feat], dim=-1)
        color = self.forward_color_net(h)

        outputs = torch.cat([color, sigma.unsqueeze(dim=-1)], -1)
        return outputs