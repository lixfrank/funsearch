import torch


def g_splitting(z):
    return 3 * (1 - z * (1 - z))**2 / (z * (1 - z))


def sort_and_remove_zero_cols(tensor):
    """
    对每行排序后去除全零列
    输入形状：(m, n)
    输出形状：(m, k) 其中k <= n
    """
    # 1. 每行降序排序
    sorted_tensor, _ = torch.sort(tensor, dim=1, descending=True)

    # 2. 找出非全零的列
    non_zero_cols_mask = ~torch.all(sorted_tensor == 0, dim=0)

    # 3. 筛选保留的列
    result = sorted_tensor[:, non_zero_cols_mask]

    return result


def sample_z_batch(energies, splitting_func, max_p, delta=0.05, batch_size=1024):
    """
    改进版采样函数
    参数：
    energies: 输入粒子能量张量，形状任意
    splitting_func: 分裂函数
    max_p: 最大概率值
    delta: 采样边界保护
    batch_size: 每批采样数量
    返回：
    z_values: 与输入energies形状相同的采样结果张量
    """
    device = energies.device
    mask = energies > 1.0  # 需要分裂的粒子掩码
    num_samples = mask.sum().item()

    # 初始化结果张量
    z_values = torch.zeros_like(energies)

    if num_samples == 0:
        return z_values

    # 仅对需要分裂的粒子进行采样
    samples = []
    while len(samples) < num_samples:
        x = torch.rand(batch_size, device=device) * (1 - 2*delta) + delta
        u = torch.rand(batch_size, device=device) * max_p
        fx = splitting_func(x)
        accepted = u <= fx
        samples.append(x[accepted])

    # 拼接并截取所需样本数
    final_samples = torch.cat(samples)[:num_samples]

    # 将采样结果填入对应位置
    z_values[mask] = final_samples.to(device)

    return z_values


def simulate_splitting_torch(initial_particles, delta=0.05):
    device = initial_particles.device
    divided_processes = len(initial_particles)

    # Calculate maximum value of splitting function
    z_sample = torch.tensor([delta], dtype=torch.float32, device=device)
    max_p = g_splitting(z_sample).item()

    particles = initial_particles.clone()

    while True:
        # Find particles that need splitting
        mask = particles >= 1.0
        if not mask.any():
            break

        particles = particles.flatten()
        # Sample z values for all particles to split
        z = sample_z_batch(particles, g_splitting, max_p, delta)

        # Calculate new particle energies
        particles = torch.stack([(1-z)*particles, z*particles], dim=1).flatten()
        particles = particles.view(divided_processes, -1)
        particles = sort_and_remove_zero_cols(particles)

    return particles
