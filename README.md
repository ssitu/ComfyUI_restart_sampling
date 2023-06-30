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

|Node|Description|
| --- | --- |
| KSampler With Restarts | Has all the inputs of a KSampler, but with an added string widget for configuring the restart segments and a widget for the scheduler for the restart segments. Not all samplers from KSampler are currently supported. <br>The format for `segments` is a sequence of comma separated arrays of ${[N_{\textrm{Restart}}, K, t_{\textrm{min}}, t_{\textrm{max}}]}$. For example, [4, 1, 19.35, 40.79], [4, 1, 1.09, 1.92], [4, 5, 0.59, 1.09], [4, 5, 0.30, 0.59], [6, 6, 0.06, 0.30] would be a valid sequence. For more information on the restart segments parameters, refer to the paper. <br>The `restart_scheduler` is used as the scheduler for the backwards process during restart intervals. The researchers used the karras scheduler for their experiments. |

---

## Problems:
- The node's progress bar in the UI will not accurately reflect the sampling progress. Use the console for accurate progress.
