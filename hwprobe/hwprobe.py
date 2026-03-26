#!/usr/bin/env python3
"""hwprobe — Hardware profiling for AI workstations."""


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


if __name__ == '__main__':
    get_cpu_info()
