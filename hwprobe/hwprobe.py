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

    print(f"CPU Model:          {model_name}")
    print(f"Sockets:            {num_sockets}")
    print(f"Cores per socket:   {cores_per_socket or logical_cores}")
    print(f"Total physical cores: {total_physical_cores}")
    print(f"Logical cores:      {logical_cores}")
    print(f"Threads per core:   {threads_per_core}")
    print(f"Max CPU MHz:        {max_mhz:.1f}")


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
        print("GPU:                nvidia-smi not found")
        return
    except subprocess.CalledProcessError:
        print("GPU:                nvidia-smi query failed")
        return

    for i, line in enumerate(result.stdout.strip().splitlines()):
        name, mem_total, mem_used, temp, power, pcie_gen, pcie_width = [
            v.strip() for v in line.split(',')
        ]
        print(f"GPU {i}:              {name}")
        print(f"  VRAM:             {mem_used} / {mem_total} MiB")
        print(f"  Temperature:      {temp} °C")
        print(f"  Power draw:       {power} W")
        print(f"  PCIe:             Gen{pcie_gen} x{pcie_width}")


def get_memory_info():
    with open('/proc/meminfo', 'r') as f:
        meminfo = {}
        for line in f:
            key, rest = line.split(':', 1)
            value = rest.strip().split()[0]
            meminfo[key.strip()] = int(value)

    def kb_to_gib(kb):
        return kb / (1024 * 1024)

    total = meminfo['MemTotal']
    available = meminfo['MemAvailable']
    swap_total = meminfo['SwapTotal']
    swap_used = swap_total - meminfo['SwapFree']
    hugepages = meminfo.get('HugePages_Total', 0)

    print(f"Total RAM:          {kb_to_gib(total):.1f} GiB")
    print(f"Available RAM:      {kb_to_gib(available):.1f} GiB")
    print(f"Swap total:         {kb_to_gib(swap_total):.1f} GiB")
    print(f"Swap used:          {kb_to_gib(swap_used):.1f} GiB")
    print(f"HugePages:          {'enabled' if hugepages > 0 else 'disabled'} ({hugepages} pages)")


def get_numa_info():
    node_dirs = sorted(glob.glob('/sys/devices/system/node/node*'))
    print(f"NUMA Nodes:         {len(node_dirs)}")

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

        mem_gib = mem_total_kb / (1024 * 1024)
        print(f"  Node {node_id}: CPUs {cpulist}, Memory {mem_gib:.1f} GiB")

    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=pci.bus_id', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return

    for i, bus_id in enumerate(result.stdout.strip().splitlines()):
        bus_id = bus_id.strip().lower()
        # nvidia-smi returns e.g. 00000000:00:1E.0, sysfs uses 0000:00:1e.0
        if len(bus_id) > 12:
            bus_id = bus_id[-12:]
        numa_path = f'/sys/bus/pci/devices/{bus_id}/numa_node'
        try:
            with open(numa_path, 'r') as f:
                numa_node = f.read().strip()
        except FileNotFoundError:
            numa_node = 'unknown'
        print(f"  GPU {i} → NUMA Node {numa_node}")


def get_storage_info():
    try:
        result = subprocess.run(
            ['lsblk', '-d', '-o', 'NAME,SIZE,TYPE,ROTA,TRAN'],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Storage:            lsblk not available")
        return

    lines = result.stdout.strip().splitlines()
    header = lines[0]
    rows = lines[1:]

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
        print(f"  {name}: {size}, {disk_type}")


if __name__ == '__main__':
    get_cpu_info()
    print()
    get_gpu_info()
    print()
    get_memory_info()
    print()
    get_numa_info()
    print()
    get_storage_info()
