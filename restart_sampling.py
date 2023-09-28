import ast
import warnings
import torch
from tqdm.auto import trange
from nodes import common_ksampler
from comfy.k_diffusion import sampling as k_diffusion_sampling
from comfy.samplers import CompVisVDenoiser
from comfy.ldm.models.diffusion.ddim import DDIMSampler
from comfy.utils import ProgressBar
from .restart_schedulers import SCHEDULER_MAPPING


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
        n_restart, k = int(n_restart), int(k)
        restart_segments = add_restart_segment(restart_segments, n_restart, k, t_min, t_max)
    return restart_segments


def segments_to_timesteps(restart_segments, model):
    timesteps = []
    for segment in restart_segments:
        t_min, t_max = model.sigma_to_t(torch.tensor(
            [segment['t_min'], segment['t_max']], device=model.log_sigmas.device))
        ts_segment = {'n': segment['n'], 'k': segment['k'], 't_min': t_min, 't_max': t_max}
        timesteps.append(ts_segment)
    return timesteps


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
                f"\n[Restart Sampling] Overwriting segment {t_min_mapping[t_min_neighbor]:.4f}, nearest neighbor of {segment['t_min']:.4f} is {t_min_neighbor}", stacklevel=2)
        t_min_mapping[t_min_neighbor] = {'n': segment['n'], 'k': segment['k'], 't_max': segment['t_max']}
    return t_min_mapping


def calc_sigmas(scheduler, n, sigma_min, sigma_max, model, device):
    return SCHEDULER_MAPPING[scheduler](model, n, sigma_min, sigma_max, device)


def calc_restart_steps(restart_segments):
    restart_steps = 0
    for segment in restart_segments.values():
        restart_steps += (segment['n'] - 1) * segment['k']
    return restart_steps


_total_steps = 0
_restart_segments = None
_restart_scheduler = None


def restart_sampling(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, restart_info, restart_scheduler, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    global _total_steps, _restart_segments, _restart_scheduler
    _restart_scheduler = restart_scheduler
    _restart_segments = prepare_restart_segments(restart_info)

    match sampler_name:
        case "ddim":
            sampler_wrapper = DDIMWrapper()
        case _:
            sampler_wrapper = KSamplerRestartWrapper(sampler_name)

    # Add the additional steps to the progress bar
    pbar_update_absolute = ProgressBar.update_absolute

    def pbar_update_absolute_wrapper(self, value, total=None, preview=None):
        pbar_update_absolute(self, value, _total_steps, preview)

    ProgressBar.update_absolute = pbar_update_absolute_wrapper

    try:
        samples = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise,
                                  disable_noise=disable_noise, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise)
    finally:
        sampler_wrapper.cleanup()
        ProgressBar.update_absolute = pbar_update_absolute
    return samples


class OneStepSampler:

    def __init__(self, model, steps, cfg, sampler, scheduler, positive, negative, latent_image, denoise):
        # Keep parameters for sampler
        self.model = model
        self.steps = steps
        self.cfg = cfg
        self.sampler = sampler
        self.scheduler = scheduler
        self.positive = positive
        self.negative = negative
        self.latent_image = latent_image
        self.denoise = denoise

        # Get the sampler function
        match sampler:
            case "ddim":
                sampler = DDIMSampler(self.model, device=self.device)
                sampler.make_schedule_timesteps(ddim_timesteps=timesteps, verbose=False)
                z_enc = sampler.stochastic_encode(latent_image, torch.tensor(
                    [len(timesteps) - 1] * noise.shape[0]).to(self.device), noise=noise, max_denoise=max_denoise)
                samples, _ = sampler.sample_custom(ddim_timesteps=timesteps,
                                                   conditioning=positive,
                                                   batch_size=noise.shape[0],
                                                   shape=noise.shape[1:],
                                                   verbose=False,
                                                   unconditional_guidance_scale=cfg,
                                                   unconditional_conditioning=negative,
                                                   eta=0.0,
                                                   x_T=z_enc,
                                                   x0=latent_image,
                                                   img_callback=ddim_callback,
                                                   denoise_function=sampling_function,
                                                   extra_args=extra_args,
                                                   mask=noise_mask,
                                                   to_zero=sigmas[-1] == 0,
                                                   end_step=sigmas.shape[0] - 1,
                                                   disable_pbar=disable_pbar)
            case _:
                sample = getattr(k_diffusion_sampling, f"sample_{sampler}")


class RestartWrapper:

    def cleanup(self):
        pass


class KSamplerRestartWrapper(RestartWrapper):

    ksampler = None

    def __init__(self, sampler_name):
        self.sample_func_name = "sample_{}".format(sampler_name)
        KSamplerRestartWrapper.ksampler = getattr(k_diffusion_sampling, self.sample_func_name)
        setattr(k_diffusion_sampling, self.sample_func_name, self.ksampler_restart_wrapper)

    def cleanup(self):
        setattr(k_diffusion_sampling, self.sample_func_name, KSamplerRestartWrapper.ksampler)

    @staticmethod
    @torch.no_grad()
    def ksampler_restart_wrapper(model, x, sigmas, extra_args=None, callback=None, disable=None):
        global _total_steps, _restart_segments, _restart_scheduler
        ksampler = KSamplerRestartWrapper.ksampler
        segments = round_restart_segments(sigmas, _restart_segments)
        _total_steps = len(sigmas) - 1 + calc_restart_steps(segments)
        step = 0

        def callback_wrapper(x):
            x["i"] = step
            if callback is not None:
                callback(x)
        with trange(_total_steps, disable=disable) as pbar:
            for i in range(len(sigmas) - 1):
                x = ksampler(model, x, torch.tensor([sigmas[i], sigmas[i + 1]],
                                                    device=x.device), extra_args, callback_wrapper, True)
                pbar.update(1)
                step += 1
                if sigmas[i + 1].item() in segments:
                    seg = segments[sigmas[i + 1].item()]
                    s_min, s_max, k, n_restart = sigmas[i + 1], seg['t_max'], seg['k'], seg['n']
                    seg_sigmas = calc_sigmas(_restart_scheduler, n_restart, s_min,
                                             s_max, model.inner_model, device=x.device)
                    for _ in range(k):
                        x += torch.randn_like(x) * (s_max ** 2 - s_min ** 2) ** 0.5
                        for j in range(n_restart - 1):
                            x = ksampler(model, x, torch.tensor(
                                [seg_sigmas[j], seg_sigmas[j + 1]], device=x.device), extra_args, callback_wrapper, True)
                            pbar.update(1)
                            step += 1
        return x


class DDIMWrapper(RestartWrapper):

    def __init__(self):
        self.__class__.sample_custom = DDIMSampler.sample_custom
        DDIMSampler.sample_custom = self.ddim_wrapper

    def cleanup(self):
        DDIMSampler.sample_custom = self.__class__.sample_custom

    @staticmethod
    @torch.no_grad()
    def ddim_wrapper(self, ddim_timesteps, conditioning=None, callback=None, img_callback=None, quantize_x0=False,
                     eta=0., mask=None, x0=None, temperature=1., noise_dropout=0., score_corrector=None,
                     corrector_kwargs=None, verbose=True, x_T=None, log_every_t=100, unconditional_guidance_scale=1.,
                     unconditional_conditioning=None, dynamic_threshold=None, ucg_schedule=None, denoise_function=None,
                     extra_args=None, to_zero=True, end_step=None, disable_pbar=False, **kwargs):
        global _total_steps, _restart_segments, _restart_scheduler
        ddim_sampler = __class__.sample_custom
        model_denoise = CompVisVDenoiser(self.model)
        segments = segments_to_timesteps(_restart_segments, model_denoise)
        segments = round_restart_segments(ddim_timesteps, segments)
        _total_steps = len(ddim_timesteps) - 1 + calc_restart_steps(segments)
        step = 0

        def callback_wrapper(pred_x0, i):
            img_callback(pred_x0, step)

        def ddim_simplified(x, timesteps, x_T=None, disable_pbar=False):
            if x_T is None:
                self.make_schedule_timesteps(ddim_timesteps=timesteps, verbose=False)
                x_T = self.stochastic_encode(x, torch.tensor(
                    [len(timesteps) - 1] * x.shape[0]).to(self.device), noise=torch.zeros_like(x), max_denoise=False)
            x, intermediates = ddim_sampler(
                self, timesteps, conditioning, callback=callback, img_callback=callback_wrapper, quantize_x0=quantize_x0,
                eta=eta, mask=mask, x0=x, temperature=temperature, noise_dropout=noise_dropout, score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs, verbose=verbose, x_T=x_T, log_every_t=log_every_t,
                unconditional_guidance_scale=unconditional_guidance_scale, unconditional_conditioning=unconditional_conditioning,
                dynamic_threshold=dynamic_threshold, ucg_schedule=ucg_schedule, denoise_function=denoise_function, extra_args=extra_args,
                to_zero=timesteps[0].item() == 0, end_step=len(timesteps) - 1, disable_pbar=disable_pbar
            )
            return x, intermediates

        with trange(_total_steps, disable=disable_pbar) as pbar:
            for i in reversed(range(len(ddim_timesteps) - 1)):
                x0, intermediates = ddim_simplified(x0, ddim_timesteps[i:i + 2], x_T=x_T, disable_pbar=True)
                x_T = None
                pbar.update(1)
                step += 1
                if ddim_timesteps[i].item() in segments:
                    seg = segments[ddim_timesteps[i].item()]
                    t_min, t_max, k, n_restart = ddim_timesteps[i], seg['t_max'], seg['k'], seg['n']
                    s_min, s_max = model_denoise.t_to_sigma(t_min), model_denoise.t_to_sigma(t_max)
                    seg_sigmas = calc_sigmas(_restart_scheduler, n_restart, s_min,
                                             s_max, model_denoise, device=x0.device)
                    for _ in range(k):
                        x0 += torch.randn_like(x0) * (s_max ** 2 - s_min ** 2) ** 0.5
                        for j in range(n_restart - 1):
                            seg_ts = model_denoise.sigma_to_t(seg_sigmas[j]).to(torch.int32)
                            seg_ts_next = model_denoise.sigma_to_t(seg_sigmas[j + 1]).to(torch.int32)
                            x0, intermediates = ddim_simplified(x0, [seg_ts_next, seg_ts], disable_pbar=True)
                            pbar.update(1)
                            step += 1
        return x0, intermediates
