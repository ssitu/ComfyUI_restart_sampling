import comfy
import torch

from . import restart_sampling as restart
from .restart_sampling import (
    DEFAULT_SEGMENTS,
    SCHEDULER_MAPPING,
    restart_sampling,
)


def get_supported_samplers():
    samplers = comfy.samplers.KSampler.SAMPLERS.copy()

    # SDE samplers cannot be used with restarts
    samplers.remove("uni_pc")
    samplers.remove("uni_pc_bh2")
    samplers.remove("dpmpp_sde")
    samplers.remove("dpmpp_sde_gpu")
    samplers.remove("dpmpp_2m_sde")
    samplers.remove("dpmpp_2m_sde_gpu")
    samplers.remove("dpmpp_3m_sde")
    samplers.remove("dpmpp_3m_sde_gpu")

    # DPM fast and adaptive go by their own schedules, restarts could be done but it won't follow the algorithm described in the paper.
    samplers.remove("dpm_fast")
    samplers.remove("dpm_adaptive")
    return samplers


def get_supported_restart_schedulers():
    return list(SCHEDULER_MAPPING.keys())


class KRestartSamplerSimple:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (get_supported_samplers(),),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "segments": ("STRING", {"default": "default", "multiline": False}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(
        self,
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise,
        segments,
    ):
        return restart_sampling(
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            segments,
            scheduler,
            denoise=denoise,
        )


class KRestartSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (get_supported_samplers(),),
                "scheduler": (tuple(SCHEDULER_MAPPING.keys()),),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "segments": (
                    "STRING",
                    {"default": DEFAULT_SEGMENTS, "multiline": False},
                ),
                "restart_scheduler": (get_supported_restart_schedulers(),),
                "chunked_mode": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(
        self,
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise,
        segments,
        restart_scheduler,
        chunked_mode=True,
    ):
        return restart_sampling(
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            segments,
            restart_scheduler,
            denoise=denoise,
            chunked_mode=chunked_mode,
        )


class KRestartSamplerAdv:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "add_noise": (["enable", "disable"],),
                "noise_seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (get_supported_samplers(),),
                "scheduler": (tuple(SCHEDULER_MAPPING.keys()),),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (["disable", "enable"],),
                "segments": (
                    "STRING",
                    {"default": DEFAULT_SEGMENTS, "multiline": False},
                ),
                "restart_scheduler": (get_supported_restart_schedulers(),),
                "chunked_mode": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(
        self,
        model,
        add_noise,
        noise_seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        start_at_step,
        end_at_step,
        return_with_leftover_noise,
        segments,
        restart_scheduler,
        chunked_mode=True,
    ):
        force_full_denoise = return_with_leftover_noise != "enable"
        disable_noise = add_noise == "disable"
        return restart_sampling(
            model,
            noise_seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            segments,
            restart_scheduler,
            disable_noise=disable_noise,
            step_range=(start_at_step, end_at_step),
            force_full_denoise=force_full_denoise,
            chunked_mode=chunked_mode,
        )


class KRestartSamplerCustom:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "add_noise": (["enable", "disable"],),
                "noise_seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler": ("SAMPLER",),
                "scheduler": (tuple(SCHEDULER_MAPPING.keys()),),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (["disable", "enable"],),
                "segments": (
                    "STRING",
                    {"default": DEFAULT_SEGMENTS, "multiline": False},
                ),
                "restart_scheduler": (get_supported_restart_schedulers(),),
                "chunked_mode": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(
        self,
        model,
        add_noise,
        noise_seed,
        steps,
        cfg,
        sampler,
        scheduler,
        positive,
        negative,
        latent_image,
        start_at_step,
        end_at_step,
        return_with_leftover_noise,
        segments,
        restart_scheduler,
        chunked_mode=True,
    ):
        force_full_denoise = return_with_leftover_noise != "enable"
        disable_noise = add_noise == "disable"
        return restart_sampling(
            model,
            noise_seed,
            steps,
            cfg,
            sampler,
            scheduler,
            positive,
            negative,
            latent_image,
            segments,
            restart_scheduler,
            disable_noise=disable_noise,
            step_range=(start_at_step, end_at_step),
            force_full_denoise=force_full_denoise,
            output_only=False,
            chunked_mode=chunked_mode,
        )


class RestartScheduler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "scheduler": (tuple(SCHEDULER_MAPPING.keys()),),
                "segments": (
                    "STRING",
                    {"default": DEFAULT_SEGMENTS, "multiline": False},
                ),
                "restart_scheduler": (get_supported_restart_schedulers(),),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "sigmas_opt": ("SIGMAS",),
            },
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "go"
    CATEGORY = "sampling/custom_sampling/schedulers"

    @staticmethod
    def plan_sigmas(plan):  # noqa: ANN205
        for pi in plan:
            yield pi.sigmas
            for _ in range(pi.k):
                yield pi.restart_sigmas * -1

    def go(
        self,
        model,
        steps,
        scheduler,
        segments,
        restart_scheduler,
        denoise,
        sigmas_opt=None,
    ):
        ms = model.get_model_object("model_sampling")
        if sigmas_opt is None or len(sigmas_opt) < 2:
            total_steps = steps
            if denoise < 1.0:
                if denoise <= 0.0:
                    return (torch.FloatTensor([]),)
                total_steps = int(steps / denoise)

            sigmas = restart.calc_sigmas(
                scheduler,
                total_steps,
                float(ms.sigma_min),
                float(ms.sigma_max),
                model.model,
                "cpu",
            )
            sigmas = sigmas[-(steps + 1) :]
        else:
            sigmas = sigmas_opt
        prepared_segments = restart.prepare_restart_segments(segments, ms, sigmas)
        plan, restart_steps = restart.build_plan(
            model.model,
            prepared_segments,
            restart_scheduler,
            sigmas,
            "cpu",
        )
        if restart.VERBOSE:
            restart.explain_plan(plan, restart_steps, chunked=True)
        restart_sigmas = torch.flatten(torch.cat(tuple(self.plan_sigmas(plan))))
        print("MADE SIGMAS", restart_sigmas)
        return (restart_sigmas,)


class RestartSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler": ("SAMPLER",),
            },
        }

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "go"
    CATEGORY = "sampling/custom_sampling/samplers"

    def go(self, sampler):
        wrapped = comfy.samplers.KSAMPLER(
            lambda *args, **kwargs: self.sampler_function(sampler, *args, **kwargs),
            extra_options=sampler.extra_options,
            inpaint_options=sampler.inpaint_options,
        )
        return (wrapped,)

    @staticmethod
    @torch.no_grad()
    def sampler_function(wrapped, model, x, sigmas, *args, **kwargs):
        last_sigma = None
        chunks = []
        while len(sigmas) > 0:
            last_sigma = None
            for idx in range(len(sigmas) - 1):
                curr_sigma = sigmas[idx + 1]
                if last_sigma is None or (
                    curr_sigma.sign() == last_sigma.sign()
                    and curr_sigma.abs() < last_sigma.abs()
                ):
                    last_sigma = curr_sigma
                    continue
                break
            if idx == len(sigmas) - 2:
                chunks.append(sigmas)
                break
            chunks.append(
                sigmas[: idx + 1] * -1 if sigmas[0] < 0 else sigmas[: idx + 1],
            )
            sigmas = sigmas[idx + 1 :]
        print("CHUNKS", chunks)
        chunks = [chunk for chunk in chunks if len(chunk) > 1]

        last_normal_chunk = 0
        for idx, chunk_sigmas in enumerate(chunks):
            print(">>>", idx, chunk_sigmas, "--", last_normal_chunk)
            if idx > 0 and chunk_sigmas[0] > chunks[last_normal_chunk][-1]:
                print("NOISE", chunk_sigmas[0], chunk_sigmas[-1])
                x += (
                    torch.randn_like(x)
                    * (chunk_sigmas[0] ** 2 - chunks[last_normal_chunk][-1] ** 2) ** 0.5
                )
            else:
                last_normal_chunk = idx
            x = wrapped.sampler_function(model, x, chunk_sigmas, *args, **kwargs)
        return x


NODE_CLASS_MAPPINGS = {
    "KRestartSamplerSimple": KRestartSamplerSimple,
    "KRestartSampler": KRestartSampler,
    "KRestartSamplerAdv": KRestartSamplerAdv,
    "KRestartSamplerCustom": KRestartSamplerCustom,
    "RestartScheduler": RestartScheduler,
    "RestartSampler": RestartSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KRestartSamplerSimple": "KSampler With Restarts (Simple)",
    "KRestartSampler": "KSampler With Restarts",
    "KRestartSamplerAdv": "KSampler With Restarts (Advanced)",
    "KRestartSamplerCustom": "KSampler With Restarts (Custom)",
}
