import comfy
from .restart_sampling import restart_sampling


def get_supported_samplers():
    samplers = comfy.samplers.KSampler.SAMPLERS.copy()
    samplers.remove("ddim")
    samplers.remove("dpm_fast")
    samplers.remove("dpm_adaptive")
    samplers.remove("uni_pc")
    samplers.remove("uni_pc_bh2")
    return samplers


def get_supported_restart_schedulers():
    schedulers = comfy.samplers.KSampler.SCHEDULERS.copy()
    schedulers.remove("simple")
    schedulers.remove("ddim_uniform")
    return schedulers


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
                "segments": ("STRING", {"default": "[3,2,0.06,0.30],[3,1,0.30,0.59]", "multiline": False}),
                "restart_scheduler": (get_supported_restart_schedulers(), {"default": "karras"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, segments, restart_scheduler):
        return restart_sampling(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, segments, restart_scheduler)


NODE_CLASS_MAPPINGS = {
    "KRestartSampler": KRestartSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KRestartSampler": "KSampler With Restarts",
}
