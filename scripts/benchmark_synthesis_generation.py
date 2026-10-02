"""Short synthetic geometry benchmark; excludes model training and auditing."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
from time import perf_counter
from types import SimpleNamespace

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name, '1')

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from privacy_defenses.synthesis_direct import DeviceGeometryCache, draw_batch


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dimension', type=int, default=49*768)
    parser.add_argument('--rank', type=int, default=99)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--views', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=10)
    args = parser.parse_args()
    if any(v < 1 for v in (args.dimension, args.rank, args.batch_size, args.views, args.iterations)):
        parser.error('All sizes must be positive.')
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        parser.error('A CUDA device is required for this comparison.')
    rng = torch.Generator().manual_seed(815)
    codes = torch.randn(args.batch_size, args.dimension, generator=rng)
    ids = torch.arange(args.batch_size)
    groups = {i: dict(mean=torch.randn(args.dimension, generator=rng),
                      factor=torch.randn(args.dimension, args.rank, generator=rng) / args.rank**.5)
              for i in range(args.batch_size)}
    geometry = SimpleNamespace(labels=ids, classes={i:{} for i in ids.tolist()},
                               global_distribution={'classes':groups})
    options = dict(center_source='global_class', global_distribution='generate')
    gpu_codes = codes.to(device)
    risk = torch.linspace(0, .98, len(codes))
    gpu_risk = risk.to(device)
    cache = DeviceGeometryCache()

    def run(use_gpu):
        generator = torch.Generator().manual_seed(9173)
        source = gpu_codes if use_gpu else gpu_codes.cpu()
        outputs = [draw_batch(geometry, source, ids, gpu_risk if use_gpu else risk,
                              options, generator, cache if use_gpu else None).to(device)
                   for _ in range(args.views)]
        return outputs, generator.get_state()

    def timed(use_gpu):
        torch.cuda.synchronize(device)
        start = perf_counter()
        result = run(use_gpu)
        torch.cuda.synchronize(device)
        return (perf_counter()-start)*1000, result

    _, (expected, cpu_state) = timed(False)
    cold_ms, (actual, gpu_state) = timed(True)
    assert torch.equal(cpu_state, gpu_state)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
    max_error = max(float((a-b).abs().max()) for a,b in zip(actual, expected))
    cpu_ms, gpu_ms = [], []
    for i in range(args.iterations):
        # Alternate timing order to reduce systematic ordering bias.
        for use_gpu in ((False, True) if i % 2 == 0 else (True, False)):
            elapsed, _ = timed(use_gpu)
            (gpu_ms if use_gpu else cpu_ms).append(elapsed)
    cpu_median, gpu_median = statistics.median(cpu_ms), statistics.median(gpu_ms)
    print(json.dumps(dict(scope='synthetic fixed batch, one class per record; generation only, warm cache',
        excludes=['input embedding', 'risk scoring', 'training', 'audit', 'CSV logging'],
        device=torch.cuda.get_device_name(device), config=vars(args),
        generation_noise='covariance_factor_times_standard_normal',
        cpu_rng_state_equal=True, max_absolute_error=max_error,
        gpu_cold_cache_ms=cold_ms, cpu_median_ms=cpu_median, gpu_warm_median_ms=gpu_median,
        generation_speedup=cpu_median/gpu_median, cpu_samples_ms=cpu_ms, gpu_samples_ms=gpu_ms,
        cache=cache.summary(), caveat='No isolation from concurrent GPU workloads; not end-to-end training speedup.'),
        indent=2))


if __name__ == '__main__':
    main()
