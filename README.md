# ComfyUI_restart_sampling
Unofficial [ComfyUI](https://github.com/comfyanonymous/ComfyUI) nodes for restart sampling based on the paper "Restart Sampling for Improving Generative Processes" 

Paper: https://arxiv.org/abs/2306.14878

Repo: https://github.com/Newbeeer/diffusion_restart_sampling

## Installation

Enter the following command from the commandline starting in ComfyUI/custom_nodes/
```
git clone https://github.com/ssitu/ComfyUI_restart_sampling
```

## Usage

Nodes can be found in the node menu under `sampling`:

|Node|Image|Description|
| --- | --- | --- |
| KSampler With Restarts | ![image](https://github.com/ssitu/ComfyUI_restart_sampling/assets/57548627/7696da21-ea8c-4263-91a9-658d0f87dc47) | Has all the inputs of a KSampler, but with an added string widget for configuring the Restart segments and a widget for the scheduler for the Restart segments. Not all samplers and schedulers from KSampler are currently supported. Restart sampling is done with ODE samplers and are not supposed to be used with SDE samplers. <br>The format for `segments` is a sequence of comma separated arrays of ${[N_{\textrm{Restart}}, K, t_{\textrm{min}}, t_{\textrm{max}}]}$. For example, [4, 1, 19.35, 40.79], [4, 1, 1.09, 1.92], [4, 5, 0.59, 1.09], [4, 5, 0.30, 0.59], [6, 6, 0.06, 0.30] would be a valid sequence. Segments may overwrite each other if their $t_{\textrm{min}}$ parameters are too close to each other. Each segment will add $(N_{\textrm{Restart}} - 1) \cdot K$ steps to the sampling process. For more information on the Restart parameters, refer to the paper. <br>The `restart_scheduler` is used as the scheduler for the denoising process during restart segments. The researchers used the Karras scheduler in their experiments, but use the same scheduler as the sampler schedule in their implementation. |
| KSampler With Restarts (Simple) | | Instead of having a restart segment scheduler, segments will use the same scheduler as the KSampler scheduler. |
| KSampler With Restarts (Advanced) | | Has all the inputs for an Advanced KSampler with all the inputs for restart sampling. It should be noted that there is a possibility for invalid segments when using it to end the denoising process early or starting it late (e.g. 20 steps, start at step 0, end at step 10) and invalid segments will be ignored. An invalid segment means that the closest $t_{\textrm{min}}$ in the noise schedule is higher than the segment's $t_{\textrm{max}}$, so the segment would have restarted the denoising process at $t_{\textrm{max}}$ then try to go to a higher noise level (when it should've gone to a lower noise level near $t_{\textrm{min}}$) which will destroy the sample. |
