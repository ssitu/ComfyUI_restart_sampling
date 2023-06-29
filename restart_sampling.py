import torch
from tqdm.auto import trange
from nodes import common_ksampler
from comfy.k_diffusion import sampling as k_diffusion_sampling


def restart_sampling(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1):
    sample_func_name = "sample_{}".format(sampler_name)
    sampler = getattr(k_diffusion_sampling, sample_func_name)

    @torch.no_grad()
    def restart_wrapper(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
        extra_args = {} if extra_args is None else extra_args

        def add_restart_segment(restart_info, n_restart, k, t_min, t_max):
            if restart_info is None:
                restart_info = []
            restart_info.append({'n': n_restart, 'k': k, 't_min': t_min, 't_max': t_max})
            return restart_info

        def one_step_heun(model, x, sigma, sigma_next, callback, extra_args):
            return sampler(model, x, torch.tensor([sigma, sigma_next], device=x.device), extra_args, callback, True, s_churn, s_tmin, s_tmax, s_noise)

        def round_restart_segments(sigmas, restart_info):
            for segment in restart_info:
                segment['t_min'] = min(sigmas, key=lambda s: abs(s - segment['t_min']))

        restart_info = add_restart_segment(None, 3, 2, .06, .3)
        round_restart_segments(sigmas, restart_info)

        for i in trange(len(sigmas) - 1, disable=disable):
            x = one_step_heun(model, x, sigmas[i], sigmas[i + 1], callback, extra_args)
            seg = None
            if any(sigmas[i + 1] == (seg := segment)['t_min'] for segment in restart_info):
                s_min, s_max, k, n_restart = seg['t_min'], seg['t_max'], seg['k'], seg['n']
                for _ in range(k):
                    # Restart sampling to t_max
                    x += torch.randn_like(x) * (s_max ** 2 - s_min ** 2) ** 0.5
                    seg_sigmas = k_diffusion_sampling.get_sigmas_karras(n_restart, s_min, s_max, device=x.device)
                    for i in trange(n_restart - 1, disable=disable):
                        x = one_step_heun(model, x, seg_sigmas[i], seg_sigmas[i + 1], callback, extra_args)
        return x

    setattr(k_diffusion_sampling, sample_func_name, restart_wrapper)
    samples = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler,
                              positive, negative, latent_image, denoise=denoise)
    setattr(k_diffusion_sampling, sample_func_name, sampler)
    return samples
