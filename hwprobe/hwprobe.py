#!/usr/bin/env python3
"""hwprobe — Hardware profiling for AI workstations."""

import glob
import subprocess


def get_cpu_info():
    with open('/proc/cpuinfo', 'r') as f:
        cpuinfo = f.read()

    blocks = cpuinfo.strip().split('\n\n')

    model_name = None
    physical_ids = set()
    logical_cores = 0
    cores_per_socket = None
    max_mhz = 0.0

    for block in blocks:
        fields = {}
        for line in block.strip().splitlines():
            if ':' in line:
                key, value = line.split(':', 1)
                fields[key.strip()] = value.strip()

        if 'processor' in fields:
            logical_cores += 1

        if model_name is None and 'model name' in fields:
            model_name = fields['model name']

        if 'physical id' in fields:
            physical_ids.add(fields['physical id'])

        if cores_per_socket is None and 'cpu cores' in fields:
            cores_per_socket = int(fields['cpu cores'])

        if 'cpu MHz' in fields:
            max_mhz = max(max_mhz, float(fields['cpu MHz']))

    num_sockets = len(physical_ids) if physical_ids else 1
    total_physical_cores = (cores_per_socket or logical_cores) * num_sockets
    threads_per_core = logical_cores // total_physical_cores if total_physical_cores else 1

    return {
        'model_name': model_name,
        'sockets': num_sockets,
        'cores_per_socket': cores_per_socket or logical_cores,
        'total_physical_cores': total_physical_cores,
        'logical_cores': logical_cores,
        'threads_per_core': threads_per_core,
        'max_mhz': max_mhz,
    }


def get_gpu_info():
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=name,memory.total,memory.used,temperature.gpu,'
                'power.draw,pcie.link.gen.current,pcie.link.width.current',
                '--format=csv,noheader,nounits',
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        return []
    except subprocess.CalledProcessError:
        return []

    gpus = []
    for line in result.stdout.strip().splitlines():
        name, mem_total, mem_used, temp, power, pcie_gen, pcie_width = [
            v.strip() for v in line.split(',')
        ]
        gpus.append({
            'name': name,
            'mem_total_mib': float(mem_total),
            'mem_used_mib': float(mem_used),
            'temperature': float(temp),
            'power_draw': float(power),
            'pcie_gen': int(pcie_gen),
            'pcie_width': int(pcie_width),
        })
    return gpus


def get_memory_info():
    with open('/proc/meminfo', 'r') as f:
        meminfo = {}
        for line in f:
            key, rest = line.split(':', 1)
            value = rest.strip().split()[0]
            meminfo[key.strip()] = int(value)

    total = meminfo['MemTotal']
    available = meminfo['MemAvailable']
    swap_total = meminfo['SwapTotal']
    swap_used = swap_total - meminfo['SwapFree']
    hugepages = meminfo.get('HugePages_Total', 0)

    return {
        'total_kb': total,
        'available_kb': available,
        'swap_total_kb': swap_total,
        'swap_used_kb': swap_used,
        'hugepages_total': hugepages,
    }


def get_numa_info():
    node_dirs = sorted(glob.glob('/sys/devices/system/node/node*'))
    nodes = []

    for node_dir in node_dirs:
        node_id = node_dir.rsplit('node', 1)[-1]

        with open(f'{node_dir}/cpulist', 'r') as f:
            cpulist = f.read().strip()

        mem_total_kb = 0
        with open(f'{node_dir}/meminfo', 'r') as f:
            for line in f:
                if 'MemTotal' in line:
                    mem_total_kb = int(line.split()[-2])
                    break

        nodes.append({
            'id': node_id,
            'cpulist': cpulist,
            'mem_total_kb': mem_total_kb,
        })

    gpu_numa = []
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=pci.bus_id', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True,
        )
        for i, bus_id in enumerate(result.stdout.strip().splitlines()):
            bus_id = bus_id.strip().lower()
            if len(bus_id) > 12:
                bus_id = bus_id[-12:]
            numa_path = f'/sys/bus/pci/devices/{bus_id}/numa_node'
            try:
                with open(numa_path, 'r') as f:
                    numa_node = f.read().strip()
            except FileNotFoundError:
                numa_node = 'unknown'
            gpu_numa.append({'gpu_index': i, 'numa_node': numa_node})
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    return {'nodes': nodes, 'gpu_numa': gpu_numa}


def get_storage_info():
    try:
        result = subprocess.run(
            ['lsblk', '-d', '-o', 'NAME,SIZE,TYPE,ROTA,TRAN'],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    lines = result.stdout.strip().splitlines()
    rows = lines[1:]
    disks = []

    for row in rows:
        parts = row.split()
        if len(parts) < 4:
            continue

        name = parts[0]
        size = parts[1]
        dtype = parts[2]
        rota = parts[3]
        tran = parts[4] if len(parts) > 4 else ''

        if dtype == 'loop':
            continue

        disk_type = 'NVMe SSD' if tran == 'nvme' else ('SSD' if rota == '0' else 'HDD')
        disks.append({'name': name, 'size': size, 'type': disk_type})

    return disks


# --- Scoring engine ---
# Weights: GPU-heavy since VRAM and PCIe matter most for ML
WEIGHTS = {
    'gpu_vram':       3.0,
    'pcie':           2.0,
    'ram':            1.5,
    'cpu_cores':      1.0,
    'storage':        0.5,
    'gpu_temp':       0.5,
    'numa_alignment': 1.0,
    'hugepages':      0.5,
}


def calculate_score(cpu, gpus, memory, numa, storage):
    scores = {}
    warnings = []

    # GPU VRAM (use best GPU)
    if gpus:
        max_vram_gb = max(g['mem_total_mib'] for g in gpus) / 1024
        if max_vram_gb >= 24:
            scores['gpu_vram'] = 1.0
        elif max_vram_gb >= 12:
            scores['gpu_vram'] = 0.5
            warnings.append(f"GPU VRAM is {max_vram_gb:.0f} GB — 24+ GB recommended for training")
        else:
            scores['gpu_vram'] = 0.0
            warnings.append(f"GPU VRAM is {max_vram_gb:.0f} GB — too low for most ML workloads")
    else:
        scores['gpu_vram'] = 0.0
        warnings.append("No GPU detected")

    # PCIe generation (use best GPU)
    if gpus:
        max_pcie = max(g['pcie_gen'] for g in gpus)
        if max_pcie >= 4:
            scores['pcie'] = 1.0
        elif max_pcie == 3:
            scores['pcie'] = 0.5
            warnings.append("PCIe Gen3 — Gen4+ recommended for faster data transfer")
        else:
            scores['pcie'] = 0.0
            warnings.append(f"PCIe Gen{max_pcie} — severely bottlenecks GPU throughput")
    else:
        scores['pcie'] = 0.0

    # RAM
    ram_gib = memory['total_kb'] / (1024 * 1024)
    if ram_gib >= 64:
        scores['ram'] = 1.0
    elif ram_gib >= 32:
        scores['ram'] = 0.5
        warnings.append(f"RAM is {ram_gib:.0f} GiB — 64+ GiB recommended for large datasets")
    else:
        scores['ram'] = 0.0
        warnings.append(f"RAM is {ram_gib:.0f} GiB — insufficient for most ML training")

    # CPU cores
    cores = cpu['total_physical_cores']
    if cores >= 16:
        scores['cpu_cores'] = 1.0
    elif cores >= 8:
        scores['cpu_cores'] = 0.5
        warnings.append(f"{cores} CPU cores — 16+ recommended for data loading pipelines")
    else:
        scores['cpu_cores'] = 0.0
        warnings.append(f"{cores} CPU cores — will bottleneck data preprocessing")

    # Storage
    has_nvme = any(d['type'] == 'NVMe SSD' for d in storage)
    has_ssd = any('SSD' in d['type'] for d in storage)
    if has_nvme:
        scores['storage'] = 1.0
    elif has_ssd:
        scores['storage'] = 0.5
        warnings.append("No NVMe storage — NVMe recommended for fast data loading")
    else:
        scores['storage'] = 0.0
        warnings.append("No SSD detected — HDD will severely bottleneck training")

    # GPU temperature
    if gpus:
        max_temp = max(g['temperature'] for g in gpus)
        if max_temp < 80:
            scores['gpu_temp'] = 1.0
        elif max_temp < 90:
            scores['gpu_temp'] = 0.5
            warnings.append(f"GPU temp is {max_temp:.0f}°C — approaching thermal throttle")
        else:
            scores['gpu_temp'] = 0.0
            warnings.append(f"GPU temp is {max_temp:.0f}°C — thermal throttling likely")
    else:
        scores['gpu_temp'] = 0.0

    # NUMA alignment
    if numa['gpu_numa']:
        nodes_used = {g['numa_node'] for g in numa['gpu_numa']}
        if 'unknown' in nodes_used:
            scores['numa_alignment'] = 0.5
            warnings.append("GPU NUMA node unknown — cannot verify alignment")
        elif len(nodes_used) == 1 and len(numa['nodes']) >= 1:
            scores['numa_alignment'] = 1.0
        else:
            scores['numa_alignment'] = 0.0
            warnings.append("GPUs spread across NUMA nodes — may cause memory latency issues")
    else:
        scores['numa_alignment'] = 0.5

    # HugePages
    if memory['hugepages_total'] > 0:
        scores['hugepages'] = 1.0
    else:
        scores['hugepages'] = 0.0
        warnings.append("HugePages disabled — enable for better memory performance in ML")

    weighted_sum = sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)
    max_possible = sum(WEIGHTS.values())
    overall = (weighted_sum / max_possible) * 10

    if overall >= 7.5:
        verdict = "Ready for ML training"
    elif overall >= 5.0:
        verdict = "Good for inference"
    else:
        verdict = "Not recommended for ML workloads"

    return {
        'score': overall,
        'details': scores,
        'warnings': warnings,
        'verdict': verdict,
    }


# --- Display ---

def kb_to_gib(kb):
    return kb / (1024 * 1024)


def print_report(cpu, gpus, memory, numa, storage, score):
    print(f"CPU Model:          {cpu['model_name']}")
    print(f"Sockets:            {cpu['sockets']}")
    print(f"Cores per socket:   {cpu['cores_per_socket']}")
    print(f"Total physical cores: {cpu['total_physical_cores']}")
    print(f"Logical cores:      {cpu['logical_cores']}")
    print(f"Threads per core:   {cpu['threads_per_core']}")
    print(f"Max CPU MHz:        {cpu['max_mhz']:.1f}")

    print()
    if gpus:
        for i, g in enumerate(gpus):
            print(f"GPU {i}:              {g['name']}")
            print(f"  VRAM:             {g['mem_used_mib']:.0f} / {g['mem_total_mib']:.0f} MiB")
            print(f"  Temperature:      {g['temperature']:.0f} °C")
            print(f"  Power draw:       {g['power_draw']:.0f} W")
            print(f"  PCIe:             Gen{g['pcie_gen']} x{g['pcie_width']}")
    else:
        print("GPU:                not detected")

    print()
    print(f"Total RAM:          {kb_to_gib(memory['total_kb']):.1f} GiB")
    print(f"Available RAM:      {kb_to_gib(memory['available_kb']):.1f} GiB")
    print(f"Swap total:         {kb_to_gib(memory['swap_total_kb']):.1f} GiB")
    print(f"Swap used:          {kb_to_gib(memory['swap_used_kb']):.1f} GiB")
    print(f"HugePages:          {'enabled' if memory['hugepages_total'] > 0 else 'disabled'} ({memory['hugepages_total']} pages)")

    print()
    print(f"NUMA Nodes:         {len(numa['nodes'])}")
    for node in numa['nodes']:
        mem_gib = node['mem_total_kb'] / (1024 * 1024)
        print(f"  Node {node['id']}: CPUs {node['cpulist']}, Memory {mem_gib:.1f} GiB")
    for g in numa['gpu_numa']:
        print(f"  GPU {g['gpu_index']} → NUMA Node {g['numa_node']}")

    print()
    if storage:
        for d in storage:
            print(f"  {d['name']}: {d['size']}, {d['type']}")
    else:
        print("Storage:            not detected")

    print()
    print(f"{'=' * 45}")
    print(f"  AI READINESS SCORE: {score['score']:.1f}/10")
    print(f"  Verdict: {score['verdict']}")
    if score['warnings']:
        print()
        print("  Issues:")
        for w in score['warnings']:
            print(f"    - {w}")
    print(f"{'=' * 45}")


if __name__ == '__main__':
    cpu = get_cpu_info()
    gpus = get_gpu_info()
    memory = get_memory_info()
    numa = get_numa_info()
    storage = get_storage_info()
    score = calculate_score(cpu, gpus, memory, numa, storage)
    print_report(cpu, gpus, memory, numa, storage, score)
