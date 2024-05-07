# ComfyUI_restart_sampling
Unofficial [ComfyUI](https://github.com/comfyanonymous/ComfyUI) nodes for restart sampling based on the paper "Restart Sampling for Improving Generative Processes"

Paper: https://arxiv.org/abs/2306.14878

Repo: https://github.com/Newbeeer/diffusion_restart_sampling

This has been tested for ComfyUI for the following commit: [72508a8](https://github.com/comfyanonymous/ComfyUI/commit/72508a8d19121e2814ea4dfbce8a5311f37dcd61)

## Installation

Enter the following command from the commandline starting in ComfyUI/custom_nodes/
```
git clone https://github.com/ssitu/ComfyUI_restart_sampling
```

## Usage

The Restart sampler nodes can be found in the node menu under `sampling`.

If you set the environment variable `COMFYUI_VERBOSE_RESTART_SAMPLING` to `1`, restart sampling will dump
information about the steps it's going to run to the console.

### Nodes

|Node|Image|Description|
| --- | --- | --- |
| KSampler With Restarts | ![image](https://github.com/ssitu/ComfyUI_restart_sampling/assets/57548627/7696da21-ea8c-4263-91a9-658d0f87dc47) | Has all the inputs of a KSampler, but with an added string widget for configuring the Restart segments and a widget for the scheduler for the Restart segments. Not all samplers and schedulers from KSampler are currently supported. Restart sampling is done with ODE samplers and are not supposed to be used with SDE samplers. <br>See the [Segments](#segments) section below for information how to define segments. For more information on the Restart parameters, refer to the paper. <br>The `restart_scheduler` is used as the scheduler for the denoising process during restart segments. The researchers used the Karras scheduler in their experiments, but use the same scheduler as the sampler schedule in their implementation. |
| KSampler With Restarts (Simple) | | Instead of having a restart segment scheduler, segments will use the same scheduler as the KSampler scheduler. |
| KSampler With Restarts (Advanced) | | Has all the inputs for an Advanced KSampler with all the inputs for restart sampling. It should be noted that there is a possibility for invalid segments when using it to end the denoising process early or starting it late (e.g. 20 steps, start at step 0, end at step 10) and invalid segments will be ignored. An invalid segment means that the closest $t_{\textrm{min}}$ in the noise schedule is higher than the segment's $t_{\textrm{max}}$, so the segment would have restarted the denoising process at $t_{\textrm{max}}$ then try to go to a higher noise level (when it should've gone to a lower noise level near $t_{\textrm{min}}$) which will destroy the sample. |
| KSampler With Restarts (Custom) | | Essentially the same as `KSampler With Restarts (Advanced)` but it takes a `SAMPLER` input like the built in `SamplerCustom` node. Note that it is possible to input samplers that don't work properly or are incompatible with Restart sampling like SDE and UniPC samplers.|
| `RestartScheduler` | | For use with custom sampling: This node will output sigmas like other scheduler nodes with restart segments inserted. Must be used with `RestartSampler`. Like stand alone samplers, the node takes parameters for restart segments and schedules. You may also optionally connect sigmas to it, in which case it will use the supplied sigmas for the main schedule. **Note**: When sigmas are connected, the `steps` and `scheduler` parameters have no effect. Setting `denoise` also can't adjust the steps: it can only shorten the sigmas you pass to the node. |
| `RestartSampler` | | For use with custom sampling: Should be used in conjunction with `RestartScheduler` and takes a `SAMPLER` input. This node arranges for the restart noise to be injected at the appropriate points and delegates to the supplied sampler for actual sampling. |

### Segments

The format for `segments` is a sequence of comma separated arrays of ${[N_{\textrm{Restart}}, K, t_{\textrm{min}}, t_{\textrm{max}}]}$. For example, `[4, 1, 19.35, 40.79], [4, 1, 1.09, 1.92], [4, 5, 0.59, 1.09], [4, 5, 0.30, 0.59], [6, 6, 0.06, 0.30]` would be a valid sequence. Segments may overwrite each other if their $t_{\textrm{min}}$ parameters are too close to each other. Each segment will add $(N_{\textrm{Restart}} - 1) \cdot K$ steps to the sampling process.

Both $t_{\textrm{min}}$ and $t_{\textrm{max}}$ within a segment definition may be specified in any of the following three ways:

* A positive numeric value (i.e. `1.2`) — this will be interpreted as a sigma value.
* A negative numeric value between `-0` and `-1000` — this will be interpeted as a (positive) timestep. Timesteps will be converted to integer values so if you need to specify timestep `0` you can do something like `-0.1`.
* A quoted string percentage value followed by a percent sign (i.e. `"25%"`) — note that this refers to the percentage of sampling, not the percentage of steps that have elapsed.

You may freely mix the different formats. For example, `[2, 2, -500, "10%"], [3, 2, 5.3, -3]` would be a valid sequence. Note: Random numbers used for example only, not recommended.

**Special segment values**:

* Enter `default` by itself to use the default segment list.
* Enter `a1111` by itself to emulate A1111 WebUI's segment calculation behavior. For full emulation, enabled chunked mode, set both schedulers to `karras` and the sampler to `heun`.
* You may also enter `"default"` or `"a1111"` in place of a segment definition (note the quotes). This will insert preset segments at the point the quoted preset name appears. For example `[1,2,3,4], "default"` is the same as `[1,2,3,4], [3,2,0.06,0.30], [3,1,0.30,0.59]`.

### Chunked Mode

When chunked mode is enabled, the sampler is called with as many steps as possible up to the next segment. When disabled, the sampler
is only called with a single step at a time. Some samplers such as SDE samplers, momentum samplers, second order samplers
like dpmpp_2m use state from previous steps - when called step-by-step, this state is lost. Using chunked mode may make those
samplers more accurate.

*Note*: Using SDE or momentum samplers with restart is likely not an improvement over normal sampling.

## Visual Example

Consider the default segments of `[3,2,0.06,0.30],[3,1,0.30,0.59]`.

1. $N_{\textrm{Restart}}=3, {K}=2, t_{\textrm{min}}=0.06, t_{\textrm{max}}=0.30$ — closer to the end of sampling, will run two restarts two times.
2. $N_{\textrm{Restart}}=3, {K}=1, t_{\textrm{min}}=0.30, t_{\textrm{max}}=0.59$ — closer to the beginning of sampling, will run two restarts one time.

Running 20 steps with normal scheduling will look something like this:

```plaintext
Step   1: sigma=10.7
Step   2: sigma=8.08
Step   3: sigma=6.2
[... elided for brevity]
Step  15: sigma=0.596
Step  16: sigma=0.474
Step  17: sigma=0.356 -- [3,1,0.30,0.59] matches here.
  K=1:
    restart 1, step 18
    restart 2, step 19
Step  20: sigma=0.232
Step  21: sigma=0.0292 -- [3,2,0.06,0.30] matches here.
  K=1:
    restart 1, step 22
    restart 2, step 23
  K=2:
    restart 1, step 24
    restart 2, step 25
Step  26: sigma=0.0
```
