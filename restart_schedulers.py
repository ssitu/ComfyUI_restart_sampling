import torch
from comfy.k_diffusion import sampling as k_diffusion_sampling
# from comfy.samplers import normal_scheduler


def get_sigmas_karras(model, n, s_min, s_max, device):
    return k_diffusion_sampling.get_sigmas_karras(n, s_min, s_max, device=device)


def get_sigmas_exponential(model, n, s_min, s_max, device):
    return k_diffusion_sampling.get_sigmas_exponential(n, s_min, s_max, device=device)


def normal_scheduler(model, steps, s_min, s_max, sgm=False, floor=False):
    """
    Pulled from comfy.samplers.normal_scheduler
    """
    s = model.model_sampling
    start = s.timestep(torch.tensor(s_max))
    end = s.timestep(torch.tensor(s_min))

    if sgm:
        timesteps = torch.linspace(start, end, steps + 1)[:-1]
    else:
        timesteps = torch.linspace(start, end, steps)

    sigs = []
    for x in range(len(timesteps)):
        ts = timesteps[x]
        sigs.append(s.sigma(ts))
    sigs += [0.0]
    return torch.FloatTensor(sigs)


def get_sigmas_normal(model, n, s_min, s_max, device):
    return normal_scheduler(model.inner_model.inner_model, n, s_min, s_max).to(device)


def get_sigmas_simple(model, n, s_min, s_max, device):
    min_idx = torch.argmin(torch.abs(model.sigmas - s_min))
    max_idx = torch.argmin(torch.abs(model.sigmas - s_max))
    sigmas_slice = model.sigmas[min_idx:max_idx]
    ss = len(sigmas_slice) / n
    sigs = [float(s_max)]
    for x in range(1, n - 1):
        sigs += [float(sigmas_slice[-(1 + int(x * ss))])]
    sigs += [float(s_min), 0.0]
    return torch.tensor(sigs, device=device)


def get_sigmas_ddim_uniform(model, n, s_min, s_max, device):
    t_min, t_max = model.sigma_to_t(torch.tensor([s_min, s_max], device=device))
    ddim_timesteps = torch.linspace(t_max, t_min, n, dtype=torch.int16, device=device)
    sigs = []
    for ts in ddim_timesteps:
        if ts > 999:
            ts = 999
        sigs.append(model.t_to_sigma(ts))
    sigs += [0.0]
    return torch.tensor(sigs, device=device)


def get_sigmas_simple_test(model, n, s_min, s_max, device):
    min_idx = torch.argmin(torch.abs(model.sigmas - s_min))
    max_idx = torch.argmin(torch.abs(model.sigmas - s_max))
    sigmas_slice = model.sigmas[min_idx:max_idx]
    ss = len(sigmas_slice) / n
    sigs = []
    for x in range(n):
        sigs += [float(sigmas_slice[-(1 + int(x * ss))])]
    sigs += [0.0]
    return torch.tensor(sigs, device=device)


SCHEDULER_MAPPING = {
    "normal": get_sigmas_normal,
    "karras": get_sigmas_karras,
    "exponential": get_sigmas_exponential,
    "simple": get_sigmas_simple,
    "ddim_uniform": get_sigmas_ddim_uniform,
    "simple_test": get_sigmas_simple_test,
}
