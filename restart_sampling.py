from __future__ import annotations

import ast
import os
import warnings
from collections import namedtuple

import comfy
import latent_preview
import torch
from comfy.sample import prepare_noise, sample_custom
from comfy.samplers import KSAMPLER, sampler_object
from comfy.utils import ProgressBar
from tqdm.auto import trange

from .restart_schedulers import NORMAL_SCHEDULER_MAPPING, RESTART_SCHEDULER_MAPPING

VERBOSE = os.environ.get("COMFYUI_VERBOSE_RESTART_SAMPLING", "").strip() == "1"

DEFAULT_SEGMENTS = "[3,2,0.06,0.30],[3,1,0.30,0.59]"


def add_restart_segment(restart_segments, n_restart, k, t_min, t_max):
    if restart_segments is None:
        restart_segments = []
    restart_segments.append({"n": n_restart, "k": k, "t_min": t_min, "t_max": t_max})
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
    def get_a1111_segment():
        # Emulate A1111 WebUI's restart sampler behavior.
        steps = len(sigmas) - 1
        if steps < 20:
            # Less than 20 steps - no restarts.
            return []
        a1111_t_max = sigmas[int(torch.argmin(abs(sigmas - 2.0), dim=0))].item()
        if steps < 36:
            # Less than 36 steps - one restart with 9 steps.
            return [10, 1, 0.1, a1111_t_max]
        # Otherwise two restarts with steps // 4 steps.
        return [(steps // 4) + 1, 2, 0.1, a1111_t_max]

    restart_info = restart_info.strip().lower()
    if restart_info == "":
        # No restarts.
        return []
    restart_arrays = None
    if restart_info == "default":
        restart_info = DEFAULT_SEGMENTS
    elif restart_info == "a1111":
        restart_arrays = [get_a1111_segment()]
        if restart_arrays == [[]]:
            return []
    if restart_arrays is None:
        try:
            restart_arrays = ast.literal_eval(f"[{restart_info}]")
        except SyntaxError:
            print("Ill-formed restart segments")
            raise
    temp = []
    default_segments = ast.literal_eval(DEFAULT_SEGMENTS)
    # This phase expands any preset strings into actual 4-item restart segments.
    for idx in range(len(restart_arrays)):
        item = restart_arrays[idx]
        if not isinstance(item, str):
            temp.append(item)
            continue
        preset = item.strip().lower()
        if preset == "default":
            temp += default_segments
        elif preset == "a1111":
            temp.append(get_a1111_segment())
        else:
            raise ValueError("Ill-formed restart segment")
    restart_arrays = temp
    restart_segments = []
    # Now we build the actual restart segments.
    for arr in restart_arrays:
        if not isinstance(arr, (list, tuple)) or len(arr) != 4:
            raise ValueError("Restart segment must be a list with 4 values")
        n_restart, k, val_min, val_max = arr
        n_restart, k = int(n_restart), int(k)
        t_min = resolve_t_value(val_min, ms)
        t_max = resolve_t_value(val_max, ms)
        restart_segments = add_restart_segment(
            restart_segments,
            n_restart,
            k,
            t_min,
            t_max,
        )
    return restart_segments


def round_restart_segments(ts, restart_segments):
    """
    Map nearest timestep/sigma min to the nearest timestep/sigma to segments.
    :param ts: Timesteps or sigmas of the original denoising schedule
    :param restart_segments: Restart segments dict of the form {'t_min': t_min, 'n': n, 'k': k, 't_max': t_max}
    :return: dict of the form {nearest_t_min: {'n': n, 'k': k, 't_max': t_max}}
    """
    t_min_mapping = {}
    for segment in reversed(
        restart_segments,
    ):  # Reversed to prioritize segments to the front
        t_min_neighbor = min(ts, key=lambda ts: abs(ts - segment["t_min"])).item()
        if t_min_neighbor == ts[0]:
            warnings.warn(
                f"\n[Restart Sampling] nearest neighbor of segment t_min {segment['t_min']:.4f} is equal to the first t_min in the denoise schedule {ts[0]:.4f}, ignoring segment...",
                stacklevel=2,
            )
            continue
        if t_min_neighbor > segment["t_max"]:
            warnings.warn(
                f"\n[Restart Sampling] t_min neighbor {t_min_neighbor:.4f} is greater than t_max {segment['t_max']:.4f}, ignoring segment...",
                stacklevel=2,
            )
            continue
        if t_min_neighbor in t_min_mapping:
            warnings.warn(
                f"\n[Restart Sampling] Overwriting segment {t_min_mapping[t_min_neighbor]}, nearest neighbor of {segment['t_min']:.4f} is {t_min_neighbor:.4f}",
                stacklevel=2,
            )
        t_min_mapping[t_min_neighbor] = {
            "n": segment["n"],
            "k": segment["k"],
            "t_max": segment["t_max"],
        }
    return t_min_mapping


def calc_sigmas(
    scheduler,
    n,
    sigma_min,
    sigma_max,
    model,
    device,
    restart_segment=True,
):
    mapping = RESTART_SCHEDULER_MAPPING if restart_segment else NORMAL_SCHEDULER_MAPPING
    return mapping[scheduler](model, n, sigma_min, sigma_max, device)


def restart_sampling(
    model,
    seed,
    steps,
    cfg,
    sampler,
    scheduler,
    positive,
    negative,
    latent_image,
    restart_info,
    restart_scheduler,
    denoise=1.0,
    disable_noise=False,
    step_range=None,
    force_full_denoise=False,
    output_only=True,
    custom_noise=None,
    chunked_mode=True,
    sigmas=None,
):
    if isinstance(sampler, str):
        sampler = sampler_object(sampler)

    plan = RestartPlan(
        model,
        steps,
        scheduler,
        restart_info,
        restart_scheduler,
        denoise=denoise,
        step_range=step_range,
        force_full_denoise=force_full_denoise,
        sigmas=sigmas,
    )

    if VERBOSE:
        plan.explain(chunked_mode)

    total_steps = plan.total_steps
    sigmas = plan.sigmas().to(model.load_device)

    latent = latent_image
    latent_image = latent["samples"]

    if disable_noise:
        torch.manual_seed(
            seed,
        )  # workaround for https://github.com/comfyanonymous/ComfyUI/issues/2833
        noise = torch.zeros(
            latent_image.size(),
            dtype=latent_image.dtype,
            layout=latent_image.layout,
            device="cpu",
        )
    else:
        batch_inds = latent.get("batch_index", None)
        noise = prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    x0_output = {}
    callback = latent_preview.prepare_callback(model, plan.total_steps, x0_output)

    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    restart_options = {
        "restart_chunked": chunked_mode,
        "restart_wrapped_sampler": sampler,
        "restart_custom_noise": custom_noise,
    }

    ksampler = KSAMPLER(
        RestartSampler.sampler_function,
        extra_options=sampler.extra_options | restart_options,
        inpaint_options=sampler.inpaint_options | {},
    )

    # Add the additional steps to the progress bar
    pbar_update_absolute = ProgressBar.update_absolute

    def pbar_update_absolute_wrapper(self, value, total=None, preview=None):  # noqa: ARG001
        pbar_update_absolute(self, value, total_steps, preview)

    ProgressBar.update_absolute = pbar_update_absolute_wrapper

    try:
        samples = sample_custom(
            model,
            noise,
            cfg,
            ksampler,
            sigmas,
            positive,
            negative,
            latent_image,
            noise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )
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


# PlanItem:
# sigmas: Sigmas for normal (outside of a restart segment) sampling. They start from after the previous PlanItem's steps
#         if there is one or simply the beginning of sampling.
# k, s_min, s_max: This is the same as the restart segment definition. Set to 0 if there is no restart segment.
# restart_sigmas: Sigmas for the restart segment if it exists, otherwise None.
# Note: n_restart is not included as it can be calculated from the length of restart_sigmas.
class PlanItem(
    namedtuple(
        "PlanItem",
        ["sigmas", "k", "restart_sigmas"],
        defaults=[None, 0, None],
    ),
):
    __slots__ = ()

    def __new__(cls, *args: list, **kwargs: dict):
        obj = super().__new__(cls, *args, **kwargs)
        obj.validate()
        return obj

    def validate(self, threshold=1e-06):
        if len(self.sigmas) < 2:
            raise ValueError("PlanItem: invalid normal sigmas: too short")
        t = self.sigmas.sort(descending=True, stable=True)[0].unique_consecutive()
        if not torch.equal(self.sigmas, t):
            errstr = (
                f"PlanItem: invalid normal sigmas: out of order or contains duplicates: {self}",
            )
            raise ValueError(errstr)
        if self.k == 0:
            return
        if self.k < 0:
            raise ValueError("PlanItem: invalid negative k value")
        if len(self.restart_sigmas) < 2:
            raise ValueError("PlanItem: invalid restart sigmas: too short")
        if self.s_min >= self.s_max:
            raise ValueError("PlanItem: invalid min/max: min >= max")
        if self.sigmas[-1] - self.restart_sigmas[0] > threshold:
            raise ValueError(
                "PlanItem: invalid sigmas: last normal sigma >= first restart sigma",
            )
        if self.sigmas[-1] - self.restart_sigmas[-1] > threshold:
            errstr = (
                f"PlanItem: invalid sigmas: last restart sigma {self.restart_sigmas[-1]} < last normal sigma {self.sigmas[-1]}",
            )
            raise ValueError(errstr)
        t = self.restart_sigmas.sort(descending=True, stable=True)[
            0
        ].unique_consecutive()
        if not torch.equal(self.restart_sigmas, t):
            errstr = (
                f"PlanItem: invalid restart sigmas: out of order or contains duplicates: {self}",
            )
            raise ValueError(errstr)

    @property
    def total_steps(self):
        if self.k < 1:
            return len(self.sigmas) - 1
        return (len(self.sigmas) - 1) + (len(self.restart_sigmas) - 1) * self.k

    @property
    def s_min(self):
        return None if self.k < 1 else self.restart_sigmas[-1].item()

    @property
    def s_max(self):
        return None if self.k < 1 else self.restart_sigmas[0].item()


class RestartPlan:
    def __init__(
        self,
        model,
        steps,
        scheduler,
        restart_info,
        restart_scheduler,
        denoise=1.0,
        step_range=None,
        force_full_denoise=False,
        sigmas=None,
    ):
        if (
            denoise <= 0
            or (sigmas is None and steps < 1)
            or (sigmas is not None and len(sigmas) < 2)
        ):
            self.plan = []
            self.total_steps = 0
            return
        ms = model.get_model_object("model_sampling")

        if sigmas is None:
            effective_steps = steps if denoise > 0.9999 else int(steps / denoise)
            sigmas = calc_sigmas(
                scheduler,
                effective_steps,
                float(ms.sigma_min),
                float(ms.sigma_max),
                model.model,
                "cpu",
                restart_segment=False,
            )
        else:
            steps = effective_steps = len(sigmas) - 1
            steps = steps if denoise > 0.9999 else int(effective_steps * denoise)
            sigmas = sigmas.clone().detach().cpu()
        if effective_steps != steps:
            sigmas = sigmas[-(steps + 1) :]
        if step_range is not None:
            start_step, last_step = step_range

            if last_step < len(sigmas) - 1:
                sigmas = sigmas[: last_step + 1]
                if force_full_denoise:
                    sigmas[-1] = 0

            if start_step < len(sigmas) - 1:
                sigmas = sigmas[start_step:]

        restart_segments = prepare_restart_segments(restart_info, ms, sigmas)
        self.plan, self.total_steps = self.build_plan_items(
            model.model,
            restart_segments,
            restart_scheduler,
            sigmas,
            "cpu",
        )

    def __repr__(self) -> str:
        return f"<RestartPlan: steps={self.total_steps}, plan={self.plan}>"

    # Builds a list of PlanItems and calculates the total number of steps. See the comments for PlanItem
    # for more information about plans.
    # Returns two values: the plan and the total steps.
    @staticmethod
    @torch.no_grad()
    def build_plan_items(
        model,
        restart_segments,
        restart_scheduler,
        sigmas,
        device,
    ) -> tuple[list, int]:
        model_sigma_min = float(model.model_sampling.sigma_min)
        segments = round_restart_segments(sigmas, restart_segments)
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
            s_max, k, n_restart = seg["t_max"], seg["k"], seg["n"]
            if k < 1 or n_restart < 2:
                continue
            if s_max <= model_sigma_min:
                errstr = f"Restart: Invalid restart segment t_max {s_max:.05} <= model minimum sigma {model_sigma_min:.05}"
                raise ValueError(errstr)
            normal_sigmas = sigmas[range_start : i + 2]
            restart_sigmas = calc_sigmas(
                restart_scheduler,
                n_restart,
                max(model_sigma_min, sigmas[i + 1]),
                s_max,
                model,
                device=device,
            )
            if normal_sigmas[-1] != 0:
                restart_sigmas = restart_sigmas[:-1]
            restart_sigmas[-1] = s_min  # Force the restart segment to end at s_min.
            plan.append(PlanItem(normal_sigmas, k, restart_sigmas))
            range_start = -1
        if range_start != -1:
            # Include sigmas after the last restart segments in the plan.
            plan.append(PlanItem(sigmas[range_start:]))
        return plan, sum(pi.total_steps for pi in plan)

    def sigmas(self) -> torch.Tensor:
        # Flattens a plan into sigmas. When the first normal sigma matches the last item's
        # final sigma, we strip the first normal sigma to avoid creating duplicates.
        if not self.plan or self.total_steps < 1:
            return torch.FloatTensor([])

        def sigmas_generator():
            prev_last = None
            for pi in self.plan:
                yield pi.sigmas if prev_last != pi.sigmas[0] else pi.sigmas[1:]
                prev_last = pi.restart_sigmas[-1] if pi.k > 0 else pi.sigmas[-1]
                for _ in range(pi.k):
                    yield pi.restart_sigmas

        return torch.flatten(torch.cat(tuple(sigmas_generator())))

    # Dumps information about the plan to the console. It uses the normal plan execute
    # logic.
    def explain(self, chunked=True):
        def pretty_sigmas(sigmas):
            return ", ".join(f"{sig:.4}" for sig in sigmas.tolist())

        def dump_steps(step, sigmas, restart=0):
            rlabel = f"R{restart:>3}" if restart > 0 else "    "
            if chunked:
                chunk_size = len(sigmas) - 2
                step += 1
                print(
                    f"[{rlabel}] Step {step:>3}..{step+chunk_size:<3}: {pretty_sigmas(sigmas)}",
                )
                step += chunk_size
                return step
            for i in range(len(sigmas) - 1):
                step += 1
                print(f"[{rlabel}] Step {step:>3}: {pretty_sigmas(sigmas[i:i+2])}")
            return step

        print(f"** Dumping restart sampling plan (total steps {self.total_steps}):")
        step = 0
        for pi in self.plan:
            step = dump_steps(step, pi.sigmas)
            for kidx in range(pi.k):
                step = dump_steps(step, pi.restart_sigmas, kidx + 1)
        print(
            "** Plan legend: [Rn] - steps for restart #n, normal sampling steps otherwise. Ranges are inclusive.",
        )

    @staticmethod
    def self_test(
        model,
        schedules=None,
        restart_schedules=None,
        segments=None,
        min_steps=2,
        max_steps=100,
    ) -> None:
        if schedules is None:
            schedules = NORMAL_SCHEDULER_MAPPING.keys()
        if restart_schedules is None:
            restart_schedules = RESTART_SCHEDULER_MAPPING.keys()
        if segments is None:
            segments = ("default", "a1111")
        for schname in schedules:
            for rschname in restart_schedules:
                for tsegs in segments:
                    print(
                        f"--- Test: {min_steps}..{max_steps} steps, schedules {schname}/{rschname}, segments {tsegs}",
                    )
                    for tsteps in range(min_steps, max_steps + 1):
                        label = f"** {tsteps:03}: {schname}, {rschname}, {tsegs}:"
                        try:
                            _plan = RestartPlan(
                                model,
                                tsteps,
                                schname,
                                tsegs,
                                rschname,
                                1.0,
                            )
                        except ValueError as err:
                            print(f"{label}\n\t!! FAIL: {err}")
                            raise
                            continue
        print("\n|| Done test")


class RestartSampler:
    @staticmethod
    def get_segment(sigmas: torch.Tensor) -> torch.Tensor:
        # A normal segment ends when we either reach the end of the list or
        # encounter a sigma higher than the previous.
        last_sigma = sigmas[0]
        for idx in range(1, len(sigmas)):
            sigma = sigmas[idx]
            if sigma > last_sigma:
                return sigmas[:idx]
            last_sigma = sigma
        return sigmas

    @classmethod
    def split_sigmas(cls, sigmas):
        # This function just splits the sigmas into chunks that are sorted descending.
        # If the first sigma of a chunk is > the last sigma of the previous chunk then this
        # is a restart segment: noising the restart uses s_min=prev_chunk[-1], s_max=chunk[0].
        # It's a generator that yields tuples of (noise_scale, chunk_sigmas).
        prev_seg = None
        while len(sigmas) > 1:
            seg = cls.get_segment(sigmas)
            sigmas = sigmas[len(seg) :]
            if prev_seg is not None and seg[0] > prev_seg[-1]:
                s_min, s_max = prev_seg[-1], seg[0]
                noise_scale = ((s_max**2 - s_min**2) ** 0.5).item()
            else:
                noise_scale = 0.0
            prev_seg = seg
            yield (noise_scale, seg)

    # Some extra explanation for a couple of these arguments:
    #
    # restart_chunked:
    # When False, the sampling function is called step-by-step with only two sigmas at a time.
    # When True, the sampling function will be called with sigmas for multiple steps at a time.
    # this means either the steps up to the next restart segment (or the end of sampling) or the steps within
    # a restart segment.
    #
    # restart_custom_noise:
    # If set to None, restart noise will just use torch.randn_like (gaussian) for noise generation. Otherwise
    # this should contain a function that takes x, sigma_min, sigma_max, seed and returns a noise sampler
    # function (which takes sigma, sigma_next) and returns a noisy tensor.
    @classmethod
    @torch.no_grad()
    def sampler_function(
        cls,
        model,
        x,
        sigmas,
        *args: list,
        restart_wrapped_sampler=None,
        restart_chunked=True,
        restart_custom_noise=None,
        callback=None,
        disable=None,
        **kwargs: dict,
    ) -> torch.Tensor:
        if not restart_wrapped_sampler:
            raise ValueError("RestartSampler: missing restart_sampler option!")

        def restart_noise(x, _s_min, _s_max, _seed):
            return lambda _s, _sn: torch.randn_like(x)

        seed = (kwargs.get("extra_args", {}) or {}).get("seed", 42)
        if restart_custom_noise is not None:
            restart_noise = restart_custom_noise

        sampler = restart_wrapped_sampler.sampler_function

        chunks = tuple(cls.split_sigmas(sigmas))
        total_steps = sum(len(chunk) - 1 for _noise, chunk in chunks)
        step = 0
        noise_count = 0
        with trange(total_steps, disable=disable) as pbar:
            last_cb_sigma = None

            def cb_wrapper(cb_state):
                nonlocal step, last_cb_sigma
                curr_sigma = cb_state.get("sigma")
                curr_sigma = (
                    curr_sigma.item()
                    if isinstance(curr_sigma, torch.Tensor)
                    else curr_sigma
                )
                if last_cb_sigma is not None and curr_sigma == last_cb_sigma:
                    # No change since last time we were called, so we won't track it as a step.
                    return
                step += 1
                pbar.update(1)
                cb_state["i"] = step
                last_cb_sigma = curr_sigma
                if callback is not None:
                    callback(cb_state)

            def do_sample(x, sigmas):
                return sampler(
                    model,
                    x,
                    sigmas,
                    *args,
                    callback=cb_wrapper,
                    disable=True,
                    **kwargs,
                )

            for noise_scale, chunk_sigmas in chunks:
                if noise_scale != 0:
                    s_min, s_max = chunk_sigmas[-1], chunk_sigmas[0]
                    x += (
                        restart_noise(x, s_min, s_max, seed + noise_count)(s_max, s_min)
                        * noise_scale
                    )
                    noise_count += 1
                if restart_chunked:
                    x = do_sample(x, chunk_sigmas)
                    continue
                for i in range(len(chunk_sigmas) - 1):
                    x = do_sample(x, chunk_sigmas[i : i + 2])
        return x
