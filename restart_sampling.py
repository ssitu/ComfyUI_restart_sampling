import ast
from collections import namedtuple
import os
import warnings
import torch
from tqdm.auto import trange
import latent_preview
import comfy
from comfy.sample import sample_custom, prepare_noise
from comfy.samplers import KSAMPLER, sampler_object
from comfy.utils import ProgressBar
from .restart_schedulers import SCHEDULER_MAPPING

VERBOSE = os.environ.get("COMFYUI_VERBOSE_RESTART_SAMPLING", "").strip() == "1"

DEFAULT_SEGMENTS = "[3,2,0.06,0.30],[3,1,0.30,0.59]"


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


def prepare_restart_segments(restart_info, ms, sigmas):
    restart_info = restart_info.strip().lower()
    if restart_info == "":
        # No restarts.
        return []
    if restart_info == "default":
        restart_info = DEFAULT_SEGMENTS
    elif restart_info == "a1111":
        # Emulate A1111 WebUI's restart sampler behavior.
        steps = len(sigmas) - 1
        if steps < 20:
            # Less than 20 steps - no restarts.
            return []
        if steps < 36:
            # Less than 36 steps - one restart with 9 steps.
            restart_info = "[10,1,0.1,0.2]"
        else:
            # Otherwise two restarts with steps // 4 steps.
            restart_info = f"[{(steps // 4) + 1}, 2, 0.1, 0.2]"
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


def restart_sampling(model, seed, steps, cfg, sampler, scheduler, positive, negative, latent_image, restart_info, restart_scheduler, denoise=1.0, disable_noise=False, step_range=None, force_full_denoise=False, output_only=True, custom_noise=None, chunked_mode=True, sigmas=None):
    if isinstance(sampler, str):
        sampler = sampler_object(sampler)

    comfy.model_management.load_models_gpu([model])
    real_model = model
    while hasattr(real_model, "model"):
        real_model = real_model.model

    effective_steps = steps if step_range is not None or denoise > 0.9999 else int(steps / denoise)
    if sigmas is None:
        sigmas = calc_sigmas(scheduler, effective_steps,
            float(real_model.model_sampling.sigma_min), float(real_model.model_sampling.sigma_max),
            real_model, model.load_device,
        )
    else:
        sigmas = sigmas.detach().clone().to(model.load_device)
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

    restart_segments = prepare_restart_segments(restart_info, real_model.model_sampling, sigmas)

    sampler_wrapper = KSamplerRestartWrapper(sampler, real_model, restart_scheduler, restart_segments, seed, custom_noise, chunked=chunked_mode)

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
        pbar_update_absolute(self, value, sampler_wrapper.total_steps, preview)

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


class PlanItem(namedtuple("PlanItem", ["sigmas", "k", "s_min", "s_max", "restart_sigmas"], defaults=[None, 0, 0., 0., None])):
    # sigmas: Sigmas for normal (outside of a restart segment) sampling. They start from after the previous PlanItem's steps
    #         if there is one or simply the beginning of sampling.
    # k, s_min, s_max: This is the same as the restart segment definition. Set to 0 if there is no restart segment.
    # restart_sigmas: Sigmas for the restart segment if it exists, otherwise None.
    # Note: n_restart is not included as it can be calculated from the length of restart_sigmas.
    __slots__ = ()

    # Execute a plan item: runs sampling on the main sigmas, handles injecting noise for restarts
    # as well as sampling the restart steps.
    # sample:  Function used sample sigmas. It takes x, a tensor with the sigmas to sample and
    #          the restart index (k) or -1 for sampling that isn't within a restart segment.
    # get_noise_sampler: Return the noise sampler for restart segment noise injection.
    #                    It takes x, and sigma_min, sigma_max (basically the same arguments as ComfyUI's
    #                    BrownianTreeNoiseSampler class init function).
    @torch.no_grad()
    def execute(self, x, sample, get_noise_sampler):
        x = sample(x, self.sigmas, -1)
        if self.k < 1 or self.restart_sigmas is None:
            return x
        noise_sampler = get_noise_sampler(x, self.s_min, self.s_max)
        for kidx in range(self.k):
            x += noise_sampler(self.restart_sigmas[0], self.restart_sigmas[-1]) * (self.s_max ** 2 - self.s_min ** 2) ** 0.5
            x = sample(x, self.restart_sigmas, kidx)
        return x


class KSamplerRestartWrapper:
    # Some extra explanation for a couple of these arguments:
    #
    # chunked:
    # When chunked is False, the sampling function is called step-by-step with only two sigmas at a time.
    # When chunked is is True, the sampling function will be called with sigmas for multiple steps at a time.
    # this means either the steps up to the next restart segment (or the end of sampling) or the steps within
    # a restart segment.
    #
    # make_noise_sampler:
    # If set to None, restart noise will just use torch.randn_like (gaussian) for noise generation. Otherwise
    # this should contain a function that takes x, sigma_min, sigma_max, seed and returns a noise sampler
    # function (which takes sigma, sigma_next) and returns a noisy tensor.
    def __init__(self, sampler, real_model, restart_scheduler, restart_segments, seed, make_noise_sampler=None, chunked=True):
        self.ksampler = sampler
        self.real_model = real_model
        self.restart_scheduler = restart_scheduler
        self.restart_segments = restart_segments
        self.total_steps = 0
        self.seed = seed
        self.make_noise_sampler = make_noise_sampler
        self.chunked = chunked

    # Builds a list of PlanItems and calculates the total number of steps. See the comments for PlanItem
    # for more information about plans.
    # Returns two values: the plan and the total steps.
    @torch.no_grad()
    def build_plan(self, sigmas, device):
        segments = round_restart_segments(sigmas, self.restart_segments)
        total_steps = len(sigmas) - 1 + calc_restart_steps(segments)
        plan = []
        range_start = -1
        for i in range(len(sigmas) - 1):
            if range_start == -1:
                # Starting a new plan item - main sigmas start at the current index of i.
                range_start = i
            s_min = sigmas[i + 1].item()
            seg = segments.get(s_min)
            if seg is None:
                continue
            s_max, k, n_restart = seg['t_max'], seg['k'], seg['n']
            seg_sigmas = calc_sigmas(self.restart_scheduler, n_restart, s_min,
                                     s_max, self.real_model, device=device)
            plan.append(PlanItem(sigmas[range_start:i+2], k, s_min, s_max, seg_sigmas[:-1]))
            range_start = -1
        if range_start != -1:
            # Include sigmas after the last restart segments in the plan.
            plan.append(PlanItem(sigmas[range_start:]))
        return plan, total_steps

    # Dumps information about the plan to the console. It uses the normal plan execute
    # logic.
    def explain_plan(self, plan, total_steps):
        print(f"** Dumping restart sampling plan (total steps {total_steps}):")
        step = 0
        last_kidx = -1
        # Instead of actually sampling, we just dump information about the steps.
        # When kidx==-1 this is a normal step, otherwise kidx==0 is the first restart,
        # kidx==1 is the second, etc.
        def do_sample(x, sigs, kidx=-1):
            nonlocal step, last_kidx
            rlabel = f"R{kidx+1:>3}" if kidx > last_kidx else "    "
            last_kidx = kidx
            if not self.chunked:
                for i in range(len(sigs)-1):
                    step += 1
                    print(f"[{rlabel}] Step {step:>3}: {sigs[i:i+2]}")
                return x
            chunk_size = len(sigs) - 2
            step += 1
            print(f"[{rlabel}] Step {step:>3}..{step+chunk_size:<3}: {sigs}")
            step += chunk_size
            return x

        # Stub function to satisfy PlanItem.execute
        def get_noise_sampler(*_args):
            return lambda *_args: 0.0

        for pi in plan:
            pi.execute(0.0, do_sample, get_noise_sampler)
        print("** Plan legend: [Rn] - steps for restart #n, normal sampling steps otherwise. Ranges are inclusive.")


    @torch.no_grad()
    def ksampler_restart_wrapper(self, model, x, sigmas, *args, extra_args=None, callback=None, disable=None, **kwargs):
        ksampler = self.ksampler
        step = 0
        seed = self.seed
        plan, self.total_steps = self.build_plan(sigmas, x.device)

        if VERBOSE:
            self.explain_plan(plan, self.total_steps)

        def noise_sampler(*_args):
            return torch.randn_like(x)

        # Passed to the PlanItem .execute method. Most of the time, self.make_noise_sampler
        # is going to be None so this is just a wrapper for torch.randn_like.
        # Otherwise we call the noise sampler factory and increment seed to ensure that restarts
        # don't all use the same noise.
        def get_noise_sampler(x, s_min, s_max):
            nonlocal seed
            if not self.make_noise_sampler:
                return noise_sampler
            result = self.make_noise_sampler(x, s_min, s_max, seed)
            seed += 1
            return result

        with trange(self.total_steps, disable=disable) as pbar:
            def callback_wrapper(x):
                nonlocal step
                step += 1
                pbar.update(1)
                x["i"] = step
                if callback is not None:
                    callback(x)

            # Convenience function for code reuse.
            def sampler_function(x, sigs):
                return ksampler.sampler_function(
                    model, x, sigs, *args, extra_args=extra_args, callback=callback_wrapper, disable=True,
                    **kwargs)

            def do_sample(x, sigs, kidx=-1):
                if isinstance(sigs, (list, tuple)):
                    sigs = torch.tensor(sigs, device=x.device)
                if self.chunked or len(sigs) < 3:
                    # If running un chunked mode or there are already 2 or less sigmas, we can just
                    # pass the sigmas to the sampling function.
                    return sampler_function(x, sigs)
                # Otherwise we call the sampling function step by step on slices of 2 sigmas.
                for i in range(len(sigs)-1):
                    x = sampler_function(x, sigs[i:i+2])
                return x

            # Execute the plan items in sequence.
            for pi in plan:
                x = pi.execute(x, do_sample, get_noise_sampler)

        return x
