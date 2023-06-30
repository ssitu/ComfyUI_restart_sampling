import ast
import torch
from tqdm.auto import trange
from nodes import common_ksampler
from comfy.k_diffusion import sampling as k_diffusion_sampling


def add_restart_segment(restart_segments, n_restart, k, t_min, t_max):
    if restart_segments is None:
        restart_segments = []
    restart_segments.append({'n': n_restart, 'k': k, 't_min': t_min, 't_max': t_max})
    return restart_segments


def prepare_restart_segments(restart_info):
    try:
        restart_arrays = ast.literal_eval(f"[{restart_info}]")
    except SyntaxError as e:
        print("Ill-formed restart segments")
        raise e
    restart_segments = []
    for arr in restart_arrays:
        if len(arr) != 4:
            raise ValueError("Restart segment must have 4 values")
        n_restart, k, t_min, t_max = arr
        restart_segments = add_restart_segment(restart_segments, n_restart, k, t_min, t_max)
    return restart_segments


def round_restart_segments(sigmas, restart_segments):
    for segment in restart_segments:
        segment['t_min'] = min(sigmas, key=lambda s: abs(s - segment['t_min']))


def restart_sampling(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, restart_info):
    sample_func_name = "sample_{}".format(sampler_name)
    sampler = getattr(k_diffusion_sampling, sample_func_name)
    restart_segments = prepare_restart_segments(restart_info)

    @torch.no_grad()
    def restart_wrapper(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
        extra_args = {} if extra_args is None else extra_args

        round_restart_segments(sigmas, restart_segments)

        for i in trange(len(sigmas) - 1, disable=disable):
            x = sampler(model, x, torch.tensor([sigmas[i], sigmas[i + 1]], device=x.device),
                        extra_args, callback, True, s_churn, s_tmin, s_tmax, s_noise)
            seg = None
            if any(sigmas[i + 1] == (seg := segment)['t_min'] for segment in restart_segments):
                s_min, s_max, k, n_restart = seg['t_min'], seg['t_max'], seg['k'], seg['n']
                for _ in range(k):
                    x += torch.randn_like(x) * (s_max ** 2 - s_min ** 2) ** 0.5
                    seg_sigmas = k_diffusion_sampling.get_sigmas_karras(n_restart, s_min, s_max, device=x.device)
                    for i in trange(n_restart - 1, disable=disable):
                        x = sampler(model, x, torch.tensor(
                            [seg_sigmas[i], seg_sigmas[i + 1]], device=x.device), extra_args, callback, True, s_churn, s_tmin, s_tmax, s_noise)
        return x

    setattr(k_diffusion_sampling, sample_func_name, restart_wrapper)
    try:
        samples = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler,
                                  positive, negative, latent_image, denoise=denoise)
    finally:
        setattr(k_diffusion_sampling, sample_func_name, sampler)
    return samples
