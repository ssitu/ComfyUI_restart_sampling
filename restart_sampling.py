import ast
import torch
from tqdm.auto import trange
from nodes import common_ksampler
from comfy.k_diffusion import sampling as k_diffusion_sampling
from comfy.utils import ProgressBar


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


def calc_sigmas(scheduler, n, sigma_min, sigma_max, device):
    if scheduler == "karras":
        sigmas = k_diffusion_sampling.get_sigmas_karras(n, sigma_min, sigma_max, device=device)
    elif scheduler == "exponential":
        sigmas = k_diffusion_sampling.get_sigmas_exponential(n, sigma_min, sigma_max, device=device)
    else:
        raise ValueError("Unsupported scheduler")
    return sigmas


def calc_restart_steps(restart_segments):
    restart_steps = 0
    for segment in restart_segments:
        restart_steps += (segment['n'] - 1) * segment['k']
    return restart_steps


def restart_sampling(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, restart_info, restart_scheduler):
    sample_func_name = "sample_{}".format(sampler_name)
    sampler = getattr(k_diffusion_sampling, sample_func_name)
    restart_segments = prepare_restart_segments(restart_info)

    @torch.no_grad()
    def restart_wrapper(model, x, sigmas, extra_args=None, callback=None, disable=None):
        extra_args = {} if extra_args is None else extra_args
        round_restart_segments(sigmas, restart_segments)
        total_steps = (len(sigmas) - 1) + calc_restart_steps(restart_segments)
        step = 0

        def callback_wrapper(x):
            x["i"] = step
            if callback is not None:
                callback(x)

        with trange(total_steps, disable=disable) as pbar:
            for i in range(len(sigmas) - 1):
                x = sampler(model, x, torch.tensor([sigmas[i], sigmas[i + 1]],
                            device=x.device), extra_args, callback_wrapper, True)
                pbar.update(1)
                step += 1
                seg = None
                if any(sigmas[i + 1] == (seg := segment)['t_min'] for segment in restart_segments):
                    s_min, s_max, k, n_restart = seg['t_min'], seg['t_max'], seg['k'], seg['n']
                    seg_sigmas = calc_sigmas(restart_scheduler, n_restart, s_min, s_max, device=x.device)
                    for _ in range(k):
                        x += torch.randn_like(x) * (s_max ** 2 - s_min ** 2) ** 0.5
                        for j in range(n_restart - 1):
                            x = sampler(model, x, torch.tensor(
                                [seg_sigmas[j], seg_sigmas[j + 1]], device=x.device), extra_args, callback_wrapper, True)
                            pbar.update(1)
                            step += 1
        return x

    setattr(k_diffusion_sampling, sample_func_name, restart_wrapper)

    # Add the additional steps to the progress bar
    pbar_update_absolute = ProgressBar.update_absolute

    def pbar_update_absolute_wrapper(self, value, total=None, preview=None):
        pbar_update_absolute(self, value, total + calc_restart_steps(restart_segments), preview)

    ProgressBar.update_absolute = pbar_update_absolute_wrapper

    try:
        samples = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler,
                                  positive, negative, latent_image, denoise=denoise)
    finally:
        setattr(k_diffusion_sampling, sample_func_name, sampler)
        ProgressBar.update_absolute = pbar_update_absolute
    return samples
