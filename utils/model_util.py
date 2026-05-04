from model.anytop import AnyTop
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps

def load_model(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])

def create_model_and_diffusion_general_skeleton(args):
    model = AnyTop(**get_gmdm_args(args))
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion

def get_gmdm_args(args):
    t5_model_dim = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-3b": 1024,
        "google/flan-t5-11b": 1024,
    }
    # default args
    t5_out_dim = t5_model_dim[args.t5_name]
    njoints = 23
    nfeats = 1
    max_joints=143 #irrelevant
    feature_len=13 #irrelevant
    cond_mode = 'object_type'
    feature_len=13

    return {'njoints': njoints, 'nfeats': nfeats, 't5_out_dim': t5_out_dim,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'max_joints': max_joints, 
            'feature_len':feature_len,  'skip_t5': args.skip_t5, 'value_emb': args.value_emb, 'root_input_feats': 13}

def get_conditioned_args(args):
    base = get_gmdm_args(args)
    base.update({
        'enc_num_queries':  args.enc_num_queries,
        'enc_fsq_dims':     args.enc_fsq_dims,
        'enc_fsq_levels':   args.enc_fsq_levels,
        'enc_use_vae':      args.enc_use_vae,
        'enc_d_z':          args.enc_d_z,
        'enc_mode':         getattr(args, 'enc_mode', None),
        'z_drop_prob':      args.z_drop_prob,
        'geom_drop_prob':   args.geom_drop_prob,
        'geom_jitter_prob': args.geom_jitter_prob,
        'topo_drop_prob':   getattr(args, 'topo_drop_prob', 0.15),
        'z_norm_target':    getattr(args, 'z_norm_target', None),
        'no_rest_pe':       getattr(args, 'no_rest_pe', False),
    })
    return base


def create_conditioned_model_and_diffusion(args):
    from model.anytop_conditioned import AnyTopConditioned
    model     = AnyTopConditioned(**get_conditioned_args(args))
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion


def create_gaussian_diffusion(args):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = 100
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_fs=args.lambda_fs,
        lambda_geo=args.lambda_geo,
    )