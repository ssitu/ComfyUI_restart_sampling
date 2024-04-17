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

from .restart_schedulers import SCHEDULER_MAPPING

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


def calc_sigmas(scheduler, n, sigma_min, sigma_max, model, device):
    return SCHEDULER_MAPPING[scheduler](model, n, sigma_min, sigma_max, device)


def calc_restart_steps(restart_segments):
    restart_steps = 0
    for segment in restart_segments.values():
        restart_steps += (segment["n"] - 1) * segment["k"]
    return restart_steps


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
    plan = plan.to(model.load_device)
    ### UNCOMMENT TO RUN SELF TEST
    # plan.self_test(
    #     model,
    #     schedules=SCHEDULER_MAPPING.keys(),
    #     restart_schedules=SCHEDULER_MAPPING.keys(),
    # )
    sigmas = plan.sigmas()

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
    callback = latent_preview.prepare_callback(
        model,
        sigmas.shape[-1] - 1,
        x0_output,
    )

    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    ksampler = KSAMPLER(
        lambda *args, **kwargs: plan.sample(
            sampler,
            *args,
            restart_chunked=chunked_mode,
            restart_make_noise_sampler=custom_noise,
            restart_seed=seed,
            **kwargs,
        ),
        extra_options=sampler.extra_options | {},
        inpaint_options=sampler.inpaint_options | {},
    )

    # Add the additional steps to the progress bar
    pbar_update_absolute = ProgressBar.update_absolute

    def pbar_update_absolute_wrapper(self, value, total=None, preview=None):
        pbar_update_absolute(self, value, plan.total_steps, preview)

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
        ["sigmas", "k", "s_min", "s_max", "restart_sigmas"],
        defaults=[None, 0, 0.0, 0.0, None],
    ),
):
    __slots__ = ()

    def __new__(cls, *args: list, **kwargs: dict):
        threshold = 1e-06
        obj = super().__new__(cls, *args, **kwargs)
        if len(obj.sigmas) < 2:
            raise ValueError("PlanItem: invalid normal sigmas: too short")
        if obj.k < 1:
            return obj
        if len(obj.restart_sigmas) < 2:
            raise ValueError("PlanItem: invalid restart sigmas: too short")
        if obj.s_min >= obj.s_max:
            raise ValueError("PlanItem: invalid min/max: min >= max")
        if obj.sigmas[-1] - obj.restart_sigmas[0] > threshold:
            raise ValueError(
                "PlanItem: invalid sigmas: last normal sigma >= first restart sigma",
            )
        if obj.sigmas[-1] - obj.restart_sigmas[-1] > threshold:
            errstr = (
                f"PlanItem: invalid sigmas: last restart sigma {obj.restart_sigmas[-1]} < last normal sigma {obj.sigmas[-1]}",
            )
            raise ValueError(errstr)
        t = obj.sigmas.sort(descending=True, stable=True)[0].unique_consecutive()
        if not torch.equal(obj.sigmas, t):
            raise ValueError(
                "PlanItem: invalid normal sigmas: out of order or contains duplicates",
            )
        t = obj.restart_sigmas.sort(descending=True, stable=True)[
            0
        ].unique_consecutive()
        if not torch.equal(obj.restart_sigmas, t):
            raise ValueError(
                "PlanItem: invalid restart sigmas: out of order or contains duplicates",
            )
        return obj

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
            x += (
                noise_sampler(self.restart_sigmas[0], self.restart_sigmas[-1])
                * (self.s_max**2 - self.s_min**2) ** 0.5
            )
            x = sample(x, self.restart_sigmas, kidx)
        return x


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
        comfy.model_management.load_models_gpu([model])
        real_model = model
        while hasattr(real_model, "model"):
            real_model = real_model.model

        effective_steps = (
            steps
            if step_range is not None or denoise > 0.9999
            else int(steps / denoise)
        )
        if sigmas is None:
            sigmas = calc_sigmas(
                scheduler,
                effective_steps,
                float(real_model.model_sampling.sigma_min),
                float(real_model.model_sampling.sigma_max),
                real_model,
                "cpu",
            )
        else:
            sigmas = sigmas.detach().cpu().clone()
        if step_range is not None:
            start_step, last_step = step_range

            if last_step < (len(sigmas) - 1):
                sigmas = sigmas[: last_step + 1]
                if force_full_denoise:
                    sigmas[-1] = 0

            if start_step < (len(sigmas) - 1):
                sigmas = sigmas[start_step:]
        elif effective_steps != steps:
            sigmas = sigmas[-(steps + 1) :]

        self.plain_sigmas = sigmas

        restart_segments = prepare_restart_segments(
            restart_info,
            real_model.model_sampling,
            sigmas,
        )
        self.plan, self.total_steps = self.build_plan_items(
            model.model,
            restart_segments,
            restart_scheduler,
            sigmas,
            "cpu",
        )

    def __repr__(self) -> str:
        return f"<RestartPlan: steps={self.total_steps}, plan={self.plan}>"

    def __len__(self) -> int:
        return self.total_steps

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
        segments = round_restart_segments(sigmas, restart_segments)
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
            s_max, k, n_restart = seg["t_max"], seg["k"], seg["n"]
            if k < 1 or n_restart < 2:
                continue
            normal_sigmas = sigmas[range_start : i + 2]
            restart_sigmas = calc_sigmas(
                restart_scheduler,
                n_restart,
                s_min if restart_scheduler != "exponential" else max(s_min, 1e-08),
                s_max,
                model,
                device=device,
            )[:-1]
            # restart_sigmas[0] = s_max
            restart_sigmas[-1] = s_min
            plan.append(PlanItem(normal_sigmas, k, s_min, s_max, restart_sigmas))
            range_start = -1
        if range_start != -1:
            # Include sigmas after the last restart segments in the plan.
            plan.append(PlanItem(sigmas[range_start:]))
        return plan, total_steps

    @classmethod
    def from_sigmas(cls, sigmas, threshold=1e-06):
        def get_normal_segment(sigmas):
            # A normal segment ends when we either reach the end of the list or
            # encounter a sigma higher than the previous.
            last_sigma = sigmas[0]
            for idx in range(1, len(sigmas)):
                sigma = sigmas[idx]
                if last_sigma - sigma < threshold:
                    return sigmas[:idx]
                last_sigma = sigma
            return sigmas

        def get_restart_segment(sigmas, s_min):
            # s_min here is the last sigma of the previous normal segment. A restart segment
            # ends when:
            #   1. We reach the end of the list, or
            #   2. We hit a sigma greater or equal to the last sigma, or
            #   3. We hit a sigma less than s_min
            last_sigma = sigmas[0]
            for idx in range(1, len(sigmas)):
                sigma = sigmas[idx]
                if last_sigma - sigma < -threshold:
                    return sigmas[:idx]
                if sigma <= s_min:
                    return sigmas[: idx + 1]
                last_sigma = sigma
            raise ValueError("Unexpected end of sigmas in a restart segment")

        plain_sigmas = sigmas.detach().cpu().clone()
        plan = []
        total_steps = 0
        while len(sigmas) > 0:
            # Get the normal segment - a restart segment can never be first.
            normal_sigmas = get_normal_segment(sigmas)
            nslen = len(normal_sigmas)
            if nslen < 2:
                print(sigmas)
                raise ValueError(
                    "Encountered invalid normal segment rebuilding sigmas: too short",
                )
            sigmas = sigmas[nslen:]
            total_steps += nslen - 1
            if len(sigmas) == 0:
                # No restart segments follow the normal segment so we're done.
                plan.append(PlanItem(normal_sigmas))
                break
            # If we're here there has to be a restart segment; get it.
            restart_sigmas = get_restart_segment(sigmas, normal_sigmas[-1])
            rslen = len(restart_sigmas)
            if rslen < 2:
                raise ValueError(
                    "Encountered invalid normal segment rebuilding sigmas: too short",
                )
            sigmas = sigmas[rslen:]
            k = 1
            # The restart segment may be repeated multiple times. If so, count the
            # repeats and trim the sigmas list.
            while len(sigmas) > 0 and torch.equal(sigmas[:rslen], restart_sigmas):
                k += 1
                sigmas = sigmas[rslen:]
            total_steps += (rslen - 1) * k
            plan.append(
                PlanItem(
                    normal_sigmas,
                    k,
                    normal_sigmas[-1],
                    restart_sigmas[0],
                    restart_sigmas,
                ),
            )
        obj = cls.__new__(cls)
        obj.plan = plan
        obj.total_steps = total_steps
        obj.plain_sigmas = plain_sigmas
        return obj

    def sigmas(self) -> torch.Tensor:
        def sigmas_generator():
            for pi in self.plan:
                yield pi.sigmas.cpu()
                for _ in range(pi.k):
                    yield pi.restart_sigmas.cpu()

        return torch.flatten(torch.cat(tuple(sigmas_generator())))

    def to(self, device):
        obj = self.__class__.__new__(self.__class__)
        obj.plain_sigmas = self.plain_sigmas.to(device)
        obj.total_steps = self.total_steps
        items = obj.plan = []
        for pi in self.plan:
            sigmas = pi.sigmas.to(device)
            if pi.k < 1:
                items.append(PlanItem(sigmas))
                continue
            items.append(
                PlanItem(
                    sigmas,
                    pi.k,
                    pi.s_min,
                    pi.s_max,
                    pi.restart_sigmas.to(device),
                ),
            )
        return obj

    # Dumps information about the plan to the console. It uses the normal plan execute
    # logic.
    def explain(self, chunked=True):
        def pretty_sigmas(sigmas):
            return ", ".join(f"{sig:.4}" for sig in sigmas.tolist())

        print(f"** Dumping restart sampling plan (total steps {self.total_steps}):")
        for pi in self.plan:
            print(
                f"\n{pi.sigmas[-1].item():.04} .. {pi.sigmas[0].item():.04} ({len(pi.sigmas)})",
            )
            if pi.k > 0:
                print(
                    f"  {pi.restart_sigmas[-1].item():.04} ({pi.s_min:.04}) .. {pi.restart_sigmas[0].item():.04} ({pi.s_max:.04}): k={pi.k} ({len(pi.restart_sigmas)})",
                )
        step = 0
        last_kidx = -1

        # Instead of actually sampling, we just dump information about the steps.
        # When kidx==-1 this is a normal step, otherwise kidx==0 is the first restart,
        # kidx==1 is the second, etc.
        def do_sample(x, sigs, kidx=-1):
            nonlocal step, last_kidx
            rlabel = f"R{kidx+1:>3}" if kidx > last_kidx else "    "
            last_kidx = kidx
            if not chunked:
                for i in range(len(sigs) - 1):
                    step += 1
                    print(f"[{rlabel}] Step {step:>3}: {pretty_sigmas(sigs[i:i+2])}")
                return x
            chunk_size = len(sigs) - 2
            step += 1
            print(
                f"[{rlabel}] Step {step:>3}..{step+chunk_size:<3}: {pretty_sigmas(sigs)}",
            )
            step += chunk_size
            return x

        # Stub function to satisfy PlanItem.execute
        def get_noise_sampler(*_args: list):
            return lambda *_args: 0.0

        for pi in self.plan:
            pi.execute(0.0, do_sample, get_noise_sampler)
        print(
            "** Plan legend: [Rn] - steps for restart #n, normal sampling steps otherwise. Ranges are inclusive.",
        )

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
    @torch.no_grad()
    def sample(
        self,
        ksampler,
        model,
        x,
        _sigmas,
        *args: list,
        restart_chunked=True,
        restart_make_noise_sampler=None,
        restart_seed=None,
        extra_args=None,
        callback=None,
        disable=None,
        **kwargs: dict,
    ):
        step = 0
        if restart_seed is None:
            seed = (extra_args or {}).get("seed", 42)
        else:
            seed = restart_seed
        plan = self.plan

        if VERBOSE:
            self.explain(restart_chunked)

        def noise_sampler(*_args: list):
            return torch.randn_like(x)

        # Passed to the PlanItem .execute method. Most of the time, self.make_noise_sampler
        # is going to be None so this is just a wrapper for torch.randn_like.
        # Otherwise we call the noise sampler factory and increment seed to ensure that restarts
        # don't all use the same noise.
        def get_noise_sampler(x, s_min, s_max):
            nonlocal seed
            if not restart_make_noise_sampler:
                return noise_sampler
            result = restart_make_noise_sampler(x, s_min, s_max, seed)
            seed += 1
            return result

        with trange(self.total_steps, disable=disable) as pbar:
            last_cb_sigma = None

            def callback_wrapper(cb_state):
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

            # Convenience function for code reuse.
            def sampler_function(x, sigs):
                return ksampler.sampler_function(
                    model,
                    x,
                    sigs,
                    *args,
                    extra_args=extra_args,
                    callback=callback_wrapper,
                    disable=True,
                    **kwargs,
                )

            def do_sample(x, sigs, _kidx=-1):
                if isinstance(sigs, (list, tuple)):
                    sigs = torch.tensor(sigs, device=x.device)
                if restart_chunked or len(sigs) < 3:
                    # If running un chunked mode or there are already 2 or less sigmas, we can just
                    # pass the sigmas to the sampling function.
                    return sampler_function(x, sigs)
                # Otherwise we call the sampling function step by step on slices of 2 sigmas.
                for i in range(len(sigs) - 1):
                    x = sampler_function(x, sigs[i : i + 2])
                return x

            # Execute the plan items in sequence.
            for pi in plan:
                x = pi.execute(x, do_sample, get_noise_sampler)
        return x

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
            schedules = SCHEDULER_MAPPING.keys() - {"simple_test"}
        if restart_schedules is None:
            restart_schedules = SCHEDULER_MAPPING.keys() - {"simple_test"}
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
                            p1 = RestartPlan(
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
                        try:
                            p2 = RestartPlan.from_sigmas(p1.sigmas())
                        except ValueError:
                            print(label)
                            p1.explain(chunked=True)
                            raise
                        fail = None
                        if len(p1) != len(p2):
                            fail = "steps"
                        if not fail:
                            for idx in range(len(p1.plan)):
                                pi1, pi2 = p1.plan[idx], p2.plan[idx]
                                if not torch.equal(
                                    torch.round(pi1.sigmas, decimals=5),
                                    torch.round(pi2.sigmas, decimals=5),
                                ):
                                    fail = "normal"
                                    break
                                if pi1.k != pi2.k:
                                    fail = "k"
                                    break
                                if pi1.k < 1:
                                    continue
                                if not torch.equal(
                                    torch.round(pi1.restart_sigmas, decimals=5),
                                    torch.round(pi2.restart_sigmas, decimals=5),
                                ):
                                    fail = "restart"
                                    break

                        if fail:
                            print(label)
                            print("!!!", fail)
                            p1.explain()
                            print("====")
                            p2.explain()
                            raise ValueError("Failed rebuilding restart plan")
        print("\n|| Done test")
