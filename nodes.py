import os

import comfy

from .restart_sampling import (
    DEFAULT_SEGMENTS,
    NORMAL_SCHEDULER_MAPPING,
    RESTART_SCHEDULER_MAPPING,
    VERBOSE,
    RestartPlan,
    RestartSampler,
    restart_sampling,
)

INCLUDE_SELFTEST = (
    os.environ.get("COMFYUI_RESTART_SAMPLING_SELFTEST", "").strip() == "1"
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
    return tuple(RESTART_SCHEDULER_MAPPING.keys())


def get_supported_normal_schedulers():
    return tuple(NORMAL_SCHEDULER_MAPPING.keys())


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
                "scheduler": (get_supported_restart_schedulers(),),
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
                "scheduler": (get_supported_normal_schedulers(),),
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
                "scheduler": (get_supported_normal_schedulers(),),
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
                "scheduler": (get_supported_normal_schedulers(),),
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


class RestartSchedulerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "scheduler": (get_supported_normal_schedulers(),),
                "segments": (
                    "STRING",
                    {"default": DEFAULT_SEGMENTS, "multiline": False},
                ),
                "restart_scheduler": (get_supported_restart_schedulers(),),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
            },
            "optional": {
                "sigmas_opt": ("SIGMAS",),
            },
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "go"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def go(
        self,
        model,
        steps,
        scheduler,
        segments,
        restart_scheduler,
        denoise,
        start_at_step=0,
        end_at_step=10000,
        sigmas_opt=None,
    ):
        plan = RestartPlan(
            model,
            steps,
            scheduler,
            segments,
            restart_scheduler,
            denoise=denoise,
            step_range=(start_at_step, end_at_step),
            sigmas=sigmas_opt,
        )
        if VERBOSE:
            plan.explain(chunked=True)
        return (plan.sigmas(),)


class RestartSamplerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler": ("SAMPLER",),
                "chunked_mode": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "go"
    CATEGORY = "sampling/custom_sampling/samplers"

    def go(self, sampler, chunked_mode):
        restart_options = {
            "restart_chunked": chunked_mode,
            "restart_wrapped_sampler": sampler,
        }
        restart_sampler = comfy.samplers.KSAMPLER(
            RestartSampler.sampler_function,
            extra_options=sampler.extra_options | restart_options,
            inpaint_options=sampler.inpaint_options,
        )
        return (restart_sampler,)


NODE_CLASS_MAPPINGS = {
    "KRestartSamplerSimple": KRestartSamplerSimple,
    "KRestartSampler": KRestartSampler,
    "KRestartSamplerAdv": KRestartSamplerAdv,
    "KRestartSamplerCustom": KRestartSamplerCustom,
    "RestartScheduler": RestartSchedulerNode,
    "RestartSampler": RestartSamplerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KRestartSamplerSimple": "KSampler With Restarts (Simple)",
    "KRestartSampler": "KSampler With Restarts",
    "KRestartSamplerAdv": "KSampler With Restarts (Advanced)",
    "KRestartSamplerCustom": "KSampler With Restarts (Custom)",
}

if INCLUDE_SELFTEST:

    class RestartSelfTestNode:
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "model": ("MODEL",),
                    "enabled": ("BOOLEAN", {"default": True}),
                    "min_steps": ("INT", {"default": 2, "min": 0}),
                    "max_steps": ("INT", {"default": 100, "min": 2}),
                },
            }

        RETURN_TYPES = ("BOOLEAN",)
        FUNCTION = "go"
        CATEGORY = "sampling/custom_sampling/samplers"

        def go(self, model, enabled=True, min_steps=2, max_steps=100):
            if enabled:
                RestartPlan.self_test(model, min_steps=min_steps, max_steps=max_steps)
            return (True,)

    NODE_CLASS_MAPPINGS["RestartSelfTest"] = RestartSelfTestNode
