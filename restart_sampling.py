import ast
import torch
from tqdm.auto import trange
from nodes import common_ksampler
from comfy.k_diffusion import sampling as k_diffusion_sampling
from comfy.samplers import simple_scheduler, ddim_scheduler
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
    t_min_mapping = {}
    for segment in reversed(restart_segments):  # Reversed to prioritize segments to the front
        t_min_neighbor = min(sigmas, key=lambda s: abs(s - segment['t_min'])).item()
        t_min_mapping[t_min_neighbor] = {'n': segment['n'], 'k': segment['k'], 't_max': segment['t_max']}
    return t_min_mapping


def calc_sigmas(scheduler, n, sigma_min, sigma_max, model, device):
    match scheduler:
        case "karras":
            return k_diffusion_sampling.get_sigmas_karras(n, sigma_min, sigma_max, device=device)
        case "exponential":
            return k_diffusion_sampling.get_sigmas_exponential(n, sigma_min, sigma_max, device=device)
        case "normal":
            def get_sigmas(model, n, s_min, s_max):
                t_min, t_max = model.sigma_to_t(torch.tensor([s_min, s_max], device=device))
                t = torch.linspace(t_max, t_min, n, device=device)
                return k_diffusion_sampling.append_zero(model.t_to_sigma(t))
            return get_sigmas(model.inner_model, n, sigma_min, sigma_max)
        # case "simple":
        #     sigmas = simple_scheduler(model.inner_model, steps)
        # case "ddim_uniform":
        #     sigmas = ddim_scheduler(model.inner_model, steps)
        case _:
            raise ValueError("Unsupported scheduler")


def calc_restart_steps(restart_segments):
    restart_steps = 0
    for segment in restart_segments.values():
        restart_steps += (segment['n'] - 1) * segment['k']
    return restart_steps


def restart_sampling(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, restart_info, restart_scheduler):
    sample_func_name = "sample_{}".format(sampler_name)
    sampler = getattr(k_diffusion_sampling, sample_func_name)
    restart_segments = prepare_restart_segments(restart_info)
    total_steps = steps

    @torch.no_grad()
    def restart_wrapper(model, x, sigmas, extra_args=None, callback=None, disable=None):
        extra_args = {} if extra_args is None else extra_args
        segments = round_restart_segments(sigmas, restart_segments)
        nonlocal total_steps
        total_steps = steps + calc_restart_steps(segments)
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
                if sigmas[i + 1].item() in segments:
                    seg = segments[sigmas[i + 1].item()]
                    s_min, s_max, k, n_restart = sigmas[i + 1], seg['t_max'], seg['k'], seg['n']
                    seg_sigmas = calc_sigmas(restart_scheduler, n_restart, s_min, s_max, model, device=x.device)
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
        pbar_update_absolute(self, value, total_steps, preview)

    ProgressBar.update_absolute = pbar_update_absolute_wrapper

    try:
        samples = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler,
                                  positive, negative, latent_image, denoise=denoise)
    finally:
        setattr(k_diffusion_sampling, sample_func_name, sampler)
        ProgressBar.update_absolute = pbar_update_absolute
    return samples
