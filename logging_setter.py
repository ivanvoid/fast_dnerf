import os
from datetime import datetime

###
### Logging
###

def set_logging(args):
    # Create log dir and copy the config file
    basedir = args.basedir
    
    if args.i_embed==1:
        args.expname += "_hashXYZ"
    elif args.i_embed==0:
        args.expname += "_posXYZ"
    if args.i_embed_views==2:
        args.expname += "_sphereVIEW"
    elif args.i_embed_views==0:
        args.expname += "_posVIEW"
    
    args.expname += "_fine"+str(args.finest_res) + "_log2T"+str(args.log2_hashmap_size)
    
    args.expname += "_lr"+str(args.lrate) + "_decay"+str(args.lrate_decay)
    
    args.expname += "_RAdam"
    
    if args.sparse_loss_weight > 0:
        args.expname += "_sparse" + str(args.sparse_loss_weight)
    args.expname += "_TV" + str(args.tv_loss_weight)
    args.expname += datetime.now().strftime('_%H_%M_%d_%m_%Y')
    expname = args.expname

    # Create folders
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())
    return basedir, expname
