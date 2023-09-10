import comfy
from .restart_sampling import restart_sampling, SCHEDULER_MAPPING


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
    return samplers


def get_supported_restart_schedulers():
    return list(SCHEDULER_MAPPING.keys())

DEFAULT_SEGMENTS = "[3,2,0.06,0.30],[3,1,0.30,0.59]"


class KRestartSamplerSimple:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (get_supported_samplers(), ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "latent_image": ("LATENT", ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "segments": ("STRING", {"default": DEFAULT_SEGMENTS, "multiline": False}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, segments):
        return restart_sampling(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, segments, scheduler, denoise=denoise)


class KRestartSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (get_supported_samplers(), ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "latent_image": ("LATENT", ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "segments": ("STRING", {"default": DEFAULT_SEGMENTS, "multiline": False}),
                "restart_scheduler": (get_supported_restart_schedulers(), ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, segments, restart_scheduler):
        return restart_sampling(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, segments, restart_scheduler, denoise=denoise)


class KRestartSamplerAdv:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "add_noise": (["enable", "disable"], ),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (get_supported_samplers(), ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "latent_image": ("LATENT", ),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (["disable", "enable"], ),
                "segments": ("STRING", {"default": DEFAULT_SEGMENTS, "multiline": False}),
                "restart_scheduler": (get_supported_restart_schedulers(), ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, segments, restart_scheduler):
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        return restart_sampling(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, segments, restart_scheduler, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)


NODE_CLASS_MAPPINGS = {
    "KRestartSamplerSimple": KRestartSamplerSimple,
    "KRestartSampler": KRestartSampler,
    "KRestartSamplerAdv": KRestartSamplerAdv,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KRestartSamplerSimple": "KSampler With Restarts (Simple)",
    "KRestartSampler": "KSampler With Restarts",
    "KRestartSamplerAdv": "KSampler With Restarts (Advanced)",
}
