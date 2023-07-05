# ComfyUI_restart_sampling
Unofficial [ComfyUI](https://github.com/comfyanonymous/ComfyUI) nodes for restart sampling based on the paper "Restart Sampling for Improving Generative Processes" 
[[paper]](https://arxiv.org/abs/2306.14878) [[repo]](https://github.com/Newbeeer/diffusion_restart_sampling)

## Installation

Enter the following command from the commandline starting in ComfyUI/custom_nodes/
```
git clone https://github.com/ssitu/ComfyUI_restart_sampling
```

## Usage

Nodes can be found in the node menu under `sampling`:

|Node|Image|Description|
| --- | --- | --- |
| KSampler With Restarts | ![image](https://github.com/ssitu/ComfyUI_restart_sampling/assets/57548627/7696da21-ea8c-4263-91a9-658d0f87dc47) | Has all the inputs of a KSampler, but with an added string widget for configuring the Restart segments and a widget for the scheduler for the Restart segments. Not all samplers and schedulers from KSampler are currently supported. Restart sampling is done with ODE samplers and are not supposed to be used with SDE samplers. <br>The format for `segments` is a sequence of comma separated arrays of ${[N_{\textrm{Restart}}, K, t_{\textrm{min}}, t_{\textrm{max}}]}$. For example, [4, 1, 19.35, 40.79], [4, 1, 1.09, 1.92], [4, 5, 0.59, 1.09], [4, 5, 0.30, 0.59], [6, 6, 0.06, 0.30] would be a valid sequence. Segments may overwrite each other if their $t_{\textrm{min}}$ parameters are too close to each other. Each segment will add $(N_{\textrm{Restart}} - 1) \cdot K$ steps to the sampling process. For more information on the Restart parameters, refer to the paper. <br>The `restart_scheduler` is used as the scheduler for the backwards process during restart intervals. The researchers used the Karras scheduler in their experiments, but use the same scheduler as the sampler schedule in their implementation. |

---

## Comparisons
These images can be dragged into ComfyUI to load their workflows.
Each image is done using the Stable Diffusion v1.5 checkpoint with 18 steps using the Heun sampler and a Karras schedule. The images in the right column use a restart segment of $[N_{\textrm{Restart}}=3, K=2, t_{\textrm{min}}=0.06, t_{\textrm{max}}=0.30]$, which adds 4 steps.
| Without | With |
| --- | --- |
| ![image](./examples/heun_edm_00002_.png) | ![image](./examples/heun_edm_restarts_00002_.png) |
| ![image](./examples/heun_edm_00003_.png) | ![image](./examples/heun_edm_restarts_00003_.png) |
| ![image](./examples/heun_edm_00001_.png) | ![image](./examples/heun_edm_restarts_00001_.png) |

Image slider links:
- https://imgsli.com/MTg5NzI4
- https://imgsli.com/MTg5NzI5
- https://imgsli.com/MTg5NzI3
