import ast
import warnings
import torch
from tqdm.auto import trange
import latent_preview
import comfy
from comfy.sample import sample_custom, prepare_noise
from comfy.samplers import KSAMPLER, sampler_object
from comfy.utils import ProgressBar
from .restart_schedulers import SCHEDULER_MAPPING


def add_restart_segment(restart_segments, n_restart, k, t_min, t_max):
    if restart_segments is None:
        restart_segments = []
    restart_segments.append({'n': n_restart, 'k': k, 't_min': t_min, 't_max': t_max})
    return restart_segments


def resolve_t_value(val, ms):
    if isinstance(val, (float, int)):
        if val >= 0.0:
            return val
        if val >= -1000:
            return ms.sigma(torch.FloatTensor([abs(int(val))], device="cpu")).item()
    if isinstance(val, str) and val.endswith("%"):
        try:
            val = float(val[:-1])
            if val >= 0 and val <= 100:
                return ms.percent_to_sigma(1.0 - val / 100.0)
        except ValueError:
            pass
    raise ValueError("bad t_min or t_max value")


def prepare_restart_segments(restart_info, ms):
    try:
        restart_arrays = ast.literal_eval(f"[{restart_info}]")
    except SyntaxError as e:
        print("Ill-formed restart segments")
        raise e
    restart_segments = []
    for arr in restart_arrays:
        if len(arr) != 4:
            raise ValueError("Restart segment must have 4 values")
        n_restart, k, val_min, val_max = arr
        n_restart, k = int(n_restart), int(k)
        t_min = resolve_t_value(val_min, ms)
        t_max = resolve_t_value(val_max, ms)
        restart_segments = add_restart_segment(restart_segments, n_restart, k, t_min, t_max)
    return restart_segments


def round_restart_segments(ts, restart_segments):
    """
    Map nearest timestep/sigma min to the nearest timestep/sigma to segments.
    :param ts: Timesteps or sigmas of the original denoising schedule
    :param restart_segments: Restart segments dict of the form {'t_min': t_min, 'n': n, 'k': k, 't_max': t_max}
    :return: dict of the form {nearest_t_min: {'n': n, 'k': k, 't_max': t_max}}
    """
    t_min_mapping = {}
    for segment in reversed(restart_segments):  # Reversed to prioritize segments to the front
        t_min_neighbor = min(ts, key=lambda ts: abs(ts - segment['t_min'])).item()
        if t_min_neighbor == ts[0]:
            warnings.warn(
                f"\n[Restart Sampling] nearest neighbor of segment t_min {segment['t_min']:.4f} is equal to the first t_min in the denoise schedule {ts[0]:.4f}, ignoring segment...", stacklevel=2)
            continue
        if t_min_neighbor > segment['t_max']:
            warnings.warn(
                f"\n[Restart Sampling] t_min neighbor {t_min_neighbor:.4f} is greater than t_max {segment['t_max']:.4f}, ignoring segment...", stacklevel=2)
            continue
        if t_min_neighbor in t_min_mapping:
            warnings.warn(
                f"\n[Restart Sampling] Overwriting segment {t_min_mapping[t_min_neighbor]}, nearest neighbor of {segment['t_min']:.4f} is {t_min_neighbor:.4f}", stacklevel=2)
        t_min_mapping[t_min_neighbor] = {'n': segment['n'], 'k': segment['k'], 't_max': segment['t_max']}
    return t_min_mapping


def calc_sigmas(scheduler, n, sigma_min, sigma_max, model, device):
    return SCHEDULER_MAPPING[scheduler](model, n, sigma_min, sigma_max, device)


def calc_restart_steps(restart_segments):
    restart_steps = 0
    for segment in restart_segments.values():
        restart_steps += (segment['n'] - 1) * segment['k']
    return restart_steps


def restart_sampling(model, seed, steps, cfg, sampler, scheduler, positive, negative, latent_image, restart_info, restart_scheduler, denoise=1.0, disable_noise=False, step_range=None, force_full_denoise=False, output_only=True, custom_noise=None, noise_multiplier=1.0, chunked_mode=False):
    if isinstance(sampler, str):
        sampler = sampler_object(sampler)


    comfy.model_management.load_models_gpu([model])
    real_model = model
    while hasattr(real_model, "model"):
        real_model = real_model.model

    restart_segments = prepare_restart_segments(restart_info, real_model.model_sampling)

    effective_steps = steps if step_range is not None or denoise > 0.9999 else int(steps / denoise)
    sigmas = calc_sigmas(scheduler, effective_steps,
        float(real_model.model_sampling.sigma_min), float(real_model.model_sampling.sigma_max),
        real_model, model.load_device,
    )
    if step_range is not None:
        start_step, last_step = step_range

        if last_step < (len(sigmas) - 1):
            sigmas = sigmas[:last_step + 1]
            if force_full_denoise:
                sigmas[-1] = 0

        if start_step < (len(sigmas) - 1):
            sigmas = sigmas[start_step:]
    elif effective_steps != steps:
        sigmas = sigmas[-(steps + 1):]

    total_steps = [0] # Updated in the wrapper.
    sampler_wrapper = KSamplerRestartWrapper(sampler, real_model, restart_scheduler, restart_segments, total_steps, seed, custom_noise, chunked=chunked_mode)

    latent = latent_image
    latent_image = latent["samples"]
    if disable_noise:
        torch.manual_seed(seed) # workaround for https://github.com/comfyanonymous/ComfyUI/issues/2833
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    x0_output = {}
    callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)

    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    sampler = KSAMPLER(
        sampler_wrapper.ksampler_restart_wrapper, extra_options=sampler.extra_options | {},
        inpaint_options=sampler.inpaint_options | {},
    )


    # Add the additional steps to the progress bar
    pbar_update_absolute = ProgressBar.update_absolute

    def pbar_update_absolute_wrapper(self, value, total=None, preview=None):
        pbar_update_absolute(self, value, total_steps[0], preview)

    ProgressBar.update_absolute = pbar_update_absolute_wrapper

    try:
        samples = sample_custom(
            model, noise, cfg, sampler, sigmas, positive, negative, latent_image,
            noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
    finally:
        ProgressBar.update_absolute = pbar_update_absolute

    out = latent.copy()
    out["samples"] = samples

    if output_only:
        return (out,)

    if "x0" in x0_output:
        out_denoised = latent.copy()
        out_denoised["samples"] = model.model.process_latent_out(x0_output["x0"].cpu())
    else:
        out_denoised = out

    return (out, out_denoised)


class KSamplerRestartWrapper:

    ksampler = None

    def __init__(self, sampler, real_model, restart_scheduler, restart_segments, total_steps, seed, custom_noise=None, chunked=True):
        self.ksampler = sampler
        self.real_model = real_model
        self.restart_scheduler = restart_scheduler
        self.restart_segments = restart_segments
        self.total_steps = total_steps
        self.seed = seed
        self.custom_noise = custom_noise
        self.chunked = chunked

    @torch.no_grad()
    def build_plan(self, x, sigmas):
        segments = round_restart_segments(sigmas, self.restart_segments)
        self.total_steps[0] = len(sigmas) - 1 + calc_restart_steps(segments)
        plan = []
        range_start = -1
        for i in range(len(sigmas) - 1):
            if range_start == -1:
                range_start = i
            s_min = sigmas[i + 1].item()
            seg = segments.get(s_min)
            if seg is None:
                continue
            s_max, k, n_restart = seg['t_max'], seg['k'], seg['n']
            seg_sigmas = calc_sigmas(self.restart_scheduler, n_restart, s_min,
                                     s_max, self.real_model, device=x.device)
            plan.append((sigmas[range_start:i+2], k, s_min, s_max, seg_sigmas[:-1]))
            range_start = -1
        if range_start != -1:
            plan.append((sigmas[range_start:], 0, 0, 0, None))
        return plan

    @torch.no_grad()
    def ksampler_restart_wrapper(self, model, x, sigmas, *args, extra_args=None, callback=None, disable=None, **kwargs):
        ksampler = self.ksampler
        def noise_sampler(_s, _sn):
            return torch.randn_like(x)
        plan = self.build_plan(x, sigmas)
        step = 0

        with trange(self.total_steps[0], disable=disable) as pbar:
            def callback_wrapper(x):
                nonlocal step
                step += 1
                pbar.update(1)
                x["i"] = step
                if callback is not None:
                    callback(x)

            def do_sample(x, sigs):
                if isinstance(sigs, (list,tuple)):
                    sigs = torch.tensor(sigs, device=x.device)
                if self.chunked or len(sigs) < 3:
                    return ksampler.sampler_function(
                                model, x, sigs, *args, extra_args=extra_args, callback=callback_wrapper, disable=True,
                                **kwargs)
                for i in range(len(sigs)-1):
                    x = do_sample(x, (sigs[i], sigs[i+1])) # This only ever recurses once.
                return x

            for chunk_sigmas, k, s_min, s_max, restart_sigmas in plan:
                x = do_sample(x, chunk_sigmas)
                if restart_sigmas is None:
                    continue
                if self.custom_noise is not None:
                    noise_sampler = self.custom_noise.make_noise_sampler(x, s_min, s_max, self.seed)
                for _ in range(k):
                    x += noise_sampler(restart_sigmas[0], restart_sigmas[-1]) * (s_max ** 2 - s_min ** 2) ** 0.5
                    x = do_sample(x, restart_sigmas)

        return x
