from radam import RAdam

from run_network import run_network
from save_and_load import load_checkpoints
from run_nerf_helpers import get_embedder

from run_nerf_helpers import DirectTemporalNeRFSmall, FD_NeRF

###
### Create objects
###


def create_nerf(args, device):
    '''
    Instantiate NeRF's MLP model.
    '''

    # Embeddings 
    embeddings = create_embeddings(args)

    # Coarse model
    coarse_model = create_coarse_model(args, embeddings, device)
    models_parameters = list(coarse_model.parameters())

    # Fine model
    fine_model = create_fine_model(args, embeddings, device)
    models_parameters += list(fine_model.parameters())

    network_query_fn = lambda inputs,viewdirs,timestep,network_fn:run_network(
        inputs, 
        viewdirs,
        timestep, 
        network_fn,
        embed_fn=embeddings['point_fn'],
        embeddirs_fn=embeddings['views_fn'],
        embedtime_fn=embeddings['time_fn'],
        netchunk=args.netchunk)

    # Create optimizer
    optimizer = create_optimizer(
        args, 
        models_parameters, 
        embeddings['optimization_parameters'])

    # Load checkpoints
    start = load_checkpoints(args, coarse_model, fine_model, embeddings, optimizer)

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : fine_model,
        'N_samples' : args.N_samples,
        'network_fn' : coarse_model,
        'embed_fn': embeddings['point_fn'],
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    ###
    outputs = {
        'render_kwargs_train':render_kwargs_train, 
        'render_kwargs_test':render_kwargs_test, 
        'start':start, 
        'models_parameters':models_parameters, 
        'optimizer':optimizer,
    }
    return outputs

def create_embeddings(args):
    # Point Embedding
    point_emb_fn, point_dim = get_embedder(args, emb_type='Hash')
    
    # Time Embedding
    time_emb_fn, time_dim = get_embedder(args, 'SinCos', 1, args.multires)

    # Get learning parameters for gradients 
    if args.i_embed==1:
        optimization_parameters = list(point_emb_fn.parameters())
    
    # Views embedder
    views_dim = 0
    views_emb_fn = None
    if args.use_viewdirs:
        # if using hashed for xyz, use SH for views ??? source?
        views_emb_fn, views_dim = get_embedder(
            args, 'SinCos', 3, args.multires_views)

    output = {
        'point_fn':point_emb_fn,
        'point_dim':point_dim,
        'time_fn':time_emb_fn,
        'time_dim':time_dim,
        'views_fn':views_emb_fn,
        'views_dim':views_dim,
        'optimization_parameters':optimization_parameters
    }

    return output

def create_coarse_model(args, embeddings, device):
    coarse_model = 0

    coarse_model = FD_NeRF(
        n_layers=4,
        hidden_dim=64,
        geo_feat_dim=15,
        n_layers_color=3,
        hidden_dim_color=64,
        input_dim_points=embeddings['point_dim'], 
        input_dim_views=embeddings['views_dim'],
        input_dim_time=embeddings['time_dim'])

    # coarse_model = DirectTemporalNeRFSmall(
    #         n_layers=2,
    #         hidden_dim=64,
    #         geo_feat_dim=15,
    #         n_layers_color=3,
    #         hidden_dim_color=64,
    #         input_dim=embeddings['point_dim'], 
    #         input_dim_views=embeddings['views_dim'],
    #         input_dim_time=embeddings['time_dim'])
    coarse_model.to(device)    

    return coarse_model

def create_fine_model(args, embeddings, device):
    '''
    Creates fine model DNERF
    '''
    fine_model = 0

    fine_model = FD_NeRF(
        n_layers=2,
        hidden_dim=64,
        geo_feat_dim=15,
        n_layers_color=3,
        hidden_dim_color=64,
        input_dim_points=embeddings['point_dim'], 
        input_dim_views=embeddings['views_dim'],
        input_dim_time=embeddings['time_dim'])

    # fine_model = DirectTemporalNeRFSmall(
    #     n_layers=2,
    #     hidden_dim=64,
    #     geo_feat_dim=15,
    #     n_layers_color=3,
    #     hidden_dim_color=64,
    #     input_dim=embeddings['point_dim'], 
    #     input_dim_views=embeddings['views_dim'],
    #     input_dim_time=embeddings['time_dim'])
    fine_model.to(device)

    return fine_model

def create_optimizer(args, parameters_of_models, parameters_of_embeddings):
    optimizer = RAdam([
                        {'params': parameters_of_models, 'weight_decay': 1e-6},
                        {'params': parameters_of_embeddings, 'eps': 1e-15}
                    ], lr=args.lrate, betas=(0.9, 0.99))
    # optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    return optimizer
