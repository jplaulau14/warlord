#!/usr/bin/env python3
"""hwprobe — Hardware profiling for AI workstations."""

import glob
import socket
import subprocess

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


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
            warnings.append(
                f"PCIe Gen{max_pcie} detected — may be power-saving (idle GPU). "
                "Run under load to verify."
            )
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
        elif nodes_used == {'-1'}:
            scores['numa_alignment'] = 1.0
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

GOOD = "\u2705"
WARN = "\u26a0\ufe0f"
CRIT = "\U0001f525"

console = Console()


def kb_to_gib(kb):
    return kb / (1024 * 1024)


def _icon(value, good_thresh, warn_thresh, higher_is_better=True):
    if higher_is_better:
        if value >= good_thresh:
            return GOOD
        elif value >= warn_thresh:
            return WARN
        else:
            return CRIT
    else:
        if value < good_thresh:
            return GOOD
        elif value < warn_thresh:
            return WARN
        else:
            return CRIT


def print_report(cpu, gpus, memory, numa, storage, score):
    hostname = socket.gethostname()
    lines = []

    # CPU
    lines.append(
        f"[bold]CPU:[/bold] {cpu['model_name']} "
        f"({cpu['total_physical_cores']} cores, {cpu['sockets']} socket)"
    )

    # RAM
    ram_gib = kb_to_gib(memory['total_kb'])
    avail_gib = kb_to_gib(memory['available_kb'])
    ram_icon = _icon(ram_gib, 64, 32)
    lines.append(
        f"[bold]RAM:[/bold] {ram_gib:.1f} GiB ({avail_gib:.1f} GiB available) {ram_icon}"
    )

    # NUMA
    node_count = len(numa['nodes'])
    lines.append(
        f"[bold]NUMA:[/bold] {node_count} node{'s' if node_count != 1 else ''}"
    )

    # GPUs
    if gpus:
        for i, g in enumerate(gpus):
            vram_gb = g['mem_total_mib'] / 1024
            vram_icon = _icon(vram_gb, 24, 12)
            lines.append(
                f"[bold]GPU {i}:[/bold] {g['name']} — "
                f"{g['mem_total_mib']:.0f} MiB VRAM {vram_icon}"
            )

            pcie_icon = _icon(g['pcie_gen'], 4, 3)
            pcie_note = ""
            if g['pcie_gen'] < 3:
                pcie_note = " (idle — may power-save)"
            lines.append(
                f"  PCIe: Gen{g['pcie_gen']} x{g['pcie_width']} {pcie_icon}{pcie_note}"
            )

            temp_icon = _icon(g['temperature'], 80, 90, higher_is_better=False)
            lines.append(
                f"  Temp: {g['temperature']:.0f}°C {temp_icon}  "
                f"Power: {g['power_draw']:.0f}W"
            )
    else:
        lines.append("[bold]GPU:[/bold] not detected")

    # Storage
    if storage:
        for d in storage:
            s_icon = GOOD if 'NVMe' in d['type'] else (WARN if 'SSD' in d['type'] else CRIT)
            lines.append(
                f"[bold]Storage:[/bold] {d['name']} {d['size']} {d['type']} {s_icon}"
            )
    else:
        lines.append(f"[bold]Storage:[/bold] not detected {CRIT}")

    # HugePages
    hp_enabled = memory['hugepages_total'] > 0
    hp_icon = GOOD if hp_enabled else WARN
    lines.append(
        f"[bold]HugePages:[/bold] {'enabled' if hp_enabled else 'disabled'} {hp_icon}"
    )

    # Build hardware section
    hw_text = Text.from_markup("\n".join(lines))

    # Build score section
    score_val = score['score']
    if score_val >= 7.5:
        score_style = "bold green"
    elif score_val >= 5.0:
        score_style = "bold yellow"
    else:
        score_style = "bold red"

    score_lines = [f"[{score_style}]SCORE: {score_val:.1f}/10 — {score['verdict']}[/]"]
    if score['warnings']:
        score_lines.append("")
        for w in score['warnings']:
            score_lines.append(f"{WARN}  {w}")

    score_text = Text.from_markup("\n".join(score_lines))

    # Print panels
    console.print()
    console.print(Panel(
        hw_text,
        title=f"[bold]HWPROBE REPORT — {hostname}[/bold]",
        title_align="left",
        border_style="cyan",
        padding=(0, 1),
    ))
    console.print(Panel(
        score_text,
        title="[bold]AI READINESS[/bold]",
        title_align="left",
        border_style="cyan",
        padding=(0, 1),
    ))


if __name__ == '__main__':
    cpu = get_cpu_info()
    gpus = get_gpu_info()
    memory = get_memory_info()
    numa = get_numa_info()
    storage = get_storage_info()
    score = calculate_score(cpu, gpus, memory, numa, storage)
    print_report(cpu, gpus, memory, numa, storage, score)
