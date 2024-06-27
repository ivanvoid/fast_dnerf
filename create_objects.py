from radam import RAdam

from run_network import run_network
from save_and_load import load_checkpoints
from run_nerf_helpers import get_embedder

from run_nerf_helpers import DirectTemporalNeRFSmall

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
    optimizer = create_optimizer(args, models_parameters, embeddings['optimization_parameters'])

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

    return render_kwargs_train, render_kwargs_test, start, models_parameters, optimizer

def create_embeddings(args):
    # Hash embedder = 1 
    embed_fn, input_ch = get_embedder(args.multires, args, i=args.i_embed)
    # Cos/Sin embedder = 0
    embedtime_fn, input_ch_time = get_embedder(args.multires, 1, i=0, input_dim=1)


    # Get learning parameters for gradients 
    if args.i_embed==1:
        # hashed embedding table
        embedding_params = list(embed_fn.parameters())
    # print(embedding_params)
    

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        # if using hashed for xyz, use SH for views
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args, i=args.i_embed_views)

    output = {
        'point_fn':embed_fn,
        'point_dim':input_ch,
        'time_fn':embedtime_fn,
        'time_dim':input_ch_time,
        'views_fn':embeddirs_fn,
        'views_dim':input_ch_views,
        'optimization_parameters':embedding_params
    }

    return output

def create_coarse_model(args, embeddings, device):
    coarse_model = 0

    coarse_model = DirectTemporalNeRFSmall(
            n_layers=2,
            hidden_dim=64,
            geo_feat_dim=15,
            n_layers_color=3,
            hidden_dim_color=64,
            input_dim=embeddings['point_dim'], 
            input_dim_views=embeddings['views_dim'],
            input_dim_time=embeddings['time_dim'])
    coarse_model.to(device)    

    return coarse_model

def create_fine_model(args, embeddings, device):
    '''
    Creates fine model DNERF
    '''
    fine_model = 0

    fine_model = DirectTemporalNeRFSmall(
        n_layers=2,
        hidden_dim=64,
        geo_feat_dim=15,
        n_layers_color=3,
        hidden_dim_color=64,
        input_dim=embeddings['point_dim'], 
        input_dim_views=embeddings['views_dim'],
        input_dim_time=embeddings['time_dim'])
    fine_model.to(device)

    return fine_model

def create_optimizer(args, parameters_of_models, parameters_of_embeddings):
    optimizer = RAdam([
                        {'params': parameters_of_models, 'weight_decay': 1e-6},
                        {'params': parameters_of_embeddings, 'eps': 1e-15}
                    ], lr=args.lrate, betas=(0.9, 0.99))
    # optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    return optimizer
