'''Adaptive training that initializes only upper-bound statistics.'''

try:
    from . import train_adaptive_count_to_adaptive as base
except ImportError:
    import train_adaptive_count_to_adaptive as base

UPPER_BOUND_SUFFIXES = (
    'upper_bound_mean', 'upper_bound_count', 'upper_bound_raw_last',
    'upper_bound_raw_min', 'upper_bound_raw_max', 'upper_bound_raw_sum',
)


def load_initial_upper_bound_stats(model, checkpoint_path, device=None):
    '''Load compatible upper-bound buffers and leave parameters untouched.'''
    del device
    if not base.os.path.exists(checkpoint_path):
        raise FileNotFoundError('Initial checkpoint not found: {}'.format(checkpoint_path))

    checkpoint = base.torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict):
        for container_key in ('state_dict_ema', 'state_dict', 'net'):
            if container_key in checkpoint:
                state_dict = checkpoint[container_key]
                break
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model_state = model.state_dict()
    checkpoint_has_module = state_dict and all(k.startswith('module.') for k in state_dict)
    model_has_module = model_state and all(k.startswith('module.') for k in model_state)
    if model_has_module and not checkpoint_has_module:
        state_dict = {'module.' + k: v for k, v in state_dict.items()}
    elif checkpoint_has_module and not model_has_module:
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

    filtered = {
        k: v for k, v in state_dict.items()
        if k.endswith(UPPER_BOUND_SUFFIXES)
        and k in model_state
        and model_state[k].shape == v.shape
    }
    model.load_state_dict(filtered, strict=False)
    base._logger.info(
        'Loaded {} upper-bound statistic tensors; all other model parameters remain '
        'freshly initialized.'.format(len(filtered))
    )
    if not filtered:
        base._logger.warning('No compatible upper-bound statistics were found.')


def collect_p99_upper_bound_stats(model):
    '''Collect P99 statistics regardless of the percentile used for training.'''
    try:
        model = base.unwrap_model(model)
    except Exception:
        if hasattr(model, 'module'):
            model = model.module

    stats = base.OrderedDict()
    layer_idx = 0
    for module_name, module in model.named_modules():
        if not all(hasattr(module, attr) for attr in UPPER_BOUND_SUFFIXES):
            continue
        if not hasattr(module, 'percentile') or module.upper_bound_count.item() <= 0:
            continue

        percentile = module.percentile.reshape(-1)
        target = base.torch.tensor(0.99, device=percentile.device, dtype=percentile.dtype)
        matches = base.torch.isclose(percentile, target, rtol=1e-6, atol=1e-8)
        if not base.torch.any(matches):
            raise ValueError('P99 must be included in --spike-percentile for WandB logging')
        stat_idx = base.torch.nonzero(matches, as_tuple=False)[0, 0].item()
        prefix = 'upper_bound/{:02d}_{}_p99'.format(
            layer_idx, module_name.replace('.', '_'))
        stats[prefix + '/raw_last'] = module.upper_bound_raw_last.reshape(-1)[stat_idx].item()
        stats[prefix + '/raw_min'] = module.upper_bound_raw_min.reshape(-1)[stat_idx].item()
        stats[prefix + '/raw_max'] = module.upper_bound_raw_max.reshape(-1)[stat_idx].item()
        stats[prefix + '/raw_avg'] = (
            module.upper_bound_raw_sum.reshape(-1)[stat_idx]
            / module.upper_bound_count.clamp_min(1.0)
        ).item()
        stats[prefix + '/ema'] = module.upper_bound_mean.reshape(-1)[stat_idx].item()
        stats[prefix + '/count'] = module.upper_bound_count.item()
        layer_idx += 1

    return stats


base.load_initial_checkpoint = load_initial_upper_bound_stats
base.collect_upper_bound_stats = collect_p99_upper_bound_stats


if __name__ == '__main__':
    base.main()
