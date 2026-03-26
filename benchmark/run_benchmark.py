import torch
import json
import time
import argparse
import sys
import os
import gc
import random
import threading
import numpy as np
from pathlib import Path
import warnings
import pynvml

# ============ 1. DETERMINISM SETUP ============
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Capture determinism warnings for paper reporting instead of silently ignoring
_determinism_warnings = []
def _warn_handler(message, category, filename, lineno, file=None, line=None):
    _determinism_warnings.append(str(message))
warnings.showwarning = _warn_handler

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        # warn_only=False: raise error on non-deterministic ops so we KNOW about them
        # If your model has unavoidable non-deterministic ops, switch back to warn_only=True
        # but log the warnings — do not suppress silently.
        torch.use_deterministic_algorithms(True, warn_only=False)
    except AttributeError:
        pass

set_seed(42)

sys.path.append(str(Path(__file__).parent.parent))

from models import get_model
from utils.config import load_config_from_yaml
from peft import PeftModel

# ============ CONFIGURATION ============
CHECKPOINT_PATHS = {
    "FROZEN": "/sda/usama/production_code/frozen_checkpoints/best_model.pt",
    "LORA_ADAPTER": "/sda/usama/production_code/clip_lora_checkpoints/epoch_3",
}

# Wearable battery parameters for deployability estimation (Section 4)
WEARABLE_BATTERY_MAH   = 300      # mAh — typical small wearable (e.g. smart glasses module)
WEARABLE_BATTERY_V     = 3.7      # Volts (LiPo nominal)
WEARABLE_BATTERY_J     = WEARABLE_BATTERY_MAH * 3.6 * WEARABLE_BATTERY_V  # J = mAh*3.6*V
INFERENCE_RATE_HZ      = 10       # target inference frequency for real-time wearable use

# Stability / safety-margin constants (Section 3)
OS_JITTER_MS           = 1.0      # conservative OS scheduling jitter (ms) for Linux RT systems
DELAY_BUFFER_MS        = 2.0      # additional conservative deployment buffer (ms)

# ---- GPU index: derive from device string so we never hardcode the wrong GPU ----
def _gpu_index(device: str) -> int:
    """
    Resolve the PHYSICAL GPU index for NVML from a PyTorch device string.

    Critical: NVML bypasses CUDA_VISIBLE_DEVICES entirely — it always uses
    physical hardware indices. PyTorch, however, remaps logical indices based
    on CUDA_VISIBLE_DEVICES. If we naively pass the PyTorch logical index to
    NVML, we measure the wrong GPU.

    Example: CUDA_VISIBLE_DEVICES=2, device='cuda' (logical index 0)
        - PyTorch thinks this is cuda:0 → runs on physical GPU 2
        - NVML index 0 → physical GPU 0 (idle!)
        - Correct NVML index → 2

    We resolve this by reading CUDA_VISIBLE_DEVICES and mapping
    logical → physical ourselves.
    """
    if device == 'cpu':
        return 0

    # Parse PyTorch logical index: 'cuda' → 0, 'cuda:1' → 1
    logical_idx = int(device.split(":")[1]) if ":" in device else 0

    # Map logical → physical via CUDA_VISIBLE_DEVICES
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        physical_devices = [int(x.strip()) for x in cvd.split(",") if x.strip().isdigit()]
        if logical_idx < len(physical_devices):
            physical = physical_devices[logical_idx]
            print(f"  [NVML] CUDA_VISIBLE_DEVICES='{cvd}' → "
                  f"logical cuda:{logical_idx} = physical GPU {physical}")
            return physical

    return logical_idx

def force_cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def strict_fp16_setup(model, device):
    target = model.model if hasattr(model, 'model') else model
    for param in target.parameters():
        param.requires_grad = False
    if hasattr(model, 'half'):
        model = model.half()
    else:
        target = target.half()
    model = model.to(device)
    model.eval()
    return model

def measure_peak_memory(func, device):
    if device == 'cpu':
        return 0.0
    force_cleanup()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        func()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)

# ============================================================
#  LATENCY  (returns per-run timings list AND summary stats)
# ============================================================
def measure_latency_stats(func, runs=300, warmup=30, device='cuda', label=''):
    """
    300 runs gives a statistically stable p99 for fast functions (CLIP ~13ms).
    For slow functions (generation ~500ms+), caller should pass runs=50.
    Returns both the stats dict and the raw timings for downstream use.
    """
    for i in range(warmup):
        func()
        if device != 'cpu':
            torch.cuda.synchronize()
        if label and (i + 1) % 5 == 0:
            print(f"     warmup {i+1}/{warmup}", flush=True)

    timings = []
    for i in range(runs):
        start = time.perf_counter()
        func()
        if device != 'cpu':
            torch.cuda.synchronize()
        end = time.perf_counter()
        timings.append((end - start) * 1000)  # ms
        # Print progress every 10 runs for slow functions (helps confirm not stuck)
        if label and (i + 1) % 10 == 0:
            print(f"     [{label}] run {i+1}/{runs}  last={timings[-1]:.1f}ms", flush=True)

    timings_arr = np.array(timings)
    stats = {
        "mean_ms":  float(np.mean(timings_arr)),
        "std_ms":   float(np.std(timings_arr)),
        "min_ms":   float(np.min(timings_arr)),
        "max_ms":   float(np.max(timings_arr)),
        "p50_ms":   float(np.percentile(timings_arr, 50)),
        "p95_ms":   float(np.percentile(timings_arr, 95)),
        "p99_ms":   float(np.percentile(timings_arr, 99)),
    }
    return stats, timings

# ============================================================
#  POWER + ENERGY  (FIX: concurrent sampling thread)
# ============================================================
def measure_power_and_energy(func, gpu_index: int, runs=100, warmup=20, device='cuda'):
    """
    Correct power measurement for IEEE publication:

    Problem with original approach
    --------------------------------
    Reading power *after* func() returns captures the GPU in its post-kernel
    idle/drain state, not the active compute state.  For short kernels this
    produces a severe underestimate of actual power draw.

    Fix: concurrent sampling thread
    --------------------------------
    A background thread polls nvml at ~1 kHz *while* the GPU is executing.
    This gives true active-compute power samples.  We then compute:

        energy_J = mean_active_power_W × mean_latency_s

    Both mean_active_power_W and mean_latency_s come from the SAME set of
    runs, so they are thermally coupled (no split-measurement bias).
    """
    if device == 'cpu':
        return {"power_W": 0.0, "energy_J": 0.0, "energy_mJ": 0.0}

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

    # Warmup to reach active thermal/power state
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()

    # --- concurrent sampling ---
    power_samples  = []
    sampling_active = threading.Event()
    sampling_done   = threading.Event()

    def sampler():
        while not sampling_done.is_set():
            if sampling_active.is_set():
                try:
                    mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                    power_samples.append(mw / 1000.0)  # → Watts
                except pynvml.NVMLError:
                    pass
            time.sleep(0.001)  # 1 ms poll interval

    t = threading.Thread(target=sampler, daemon=True)
    t.start()

    run_latencies = []
    for _ in range(runs):
        sampling_active.set()
        t0 = time.perf_counter()
        func()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        sampling_active.clear()
        run_latencies.append((t1 - t0) * 1000)

    sampling_done.set()
    t.join(timeout=2.0)
    pynvml.nvmlShutdown()

    avg_power_W   = float(np.mean(power_samples)) if power_samples else 0.0
    mean_lat_s    = float(np.mean(run_latencies)) / 1000.0
    energy_J      = avg_power_W * mean_lat_s

    return {
        "power_W":   avg_power_W,
        "energy_J":  energy_J,
        "energy_mJ": energy_J * 1000.0,
    }

# ============================================================
#  WORST-CASE SAFETY MARGIN  (Section 3)
# ============================================================
def compute_safety_margin(stats: dict) -> dict:
    """
    Worst-Case Latency Safety Margin Analysis.

    Components
    ----------
    p99_ms       : empirical 99th-percentile latency
    os_jitter_ms : conservative OS scheduling jitter for Linux systems
    buffer_ms    : additional deployment safety buffer
    wcl_ms       : Worst-Case Latency = p99 + jitter + buffer

    Stability constraint: wcl_ms should be < 1000/target_Hz to guarantee
    the model can sustain real-time inference at INFERENCE_RATE_HZ.
    """
    p99    = stats["p99_ms"]
    wcl    = p99 + OS_JITTER_MS + DELAY_BUFFER_MS
    budget = 1000.0 / INFERENCE_RATE_HZ  # ms per inference slot at target Hz
    TAU_MAX_MS = 37.80  # stability bound from Padé/Routh analysis

    return {
        "p99_ms":                    p99,
        "os_jitter_ms":              OS_JITTER_MS,
        "delay_buffer_ms":           DELAY_BUFFER_MS,
        "worst_case_latency_ms":     wcl,
        "inference_budget_ms":       budget,
        "satisfies_rt_constraint":   bool(wcl <= budget),      # throughput check
        "satisfies_stability_bound": bool(wcl < TAU_MAX_MS),   # stability check  ← ADD THIS
        "stability_margin_ms":       round(TAU_MAX_MS - wcl, 2),  # ← ADD THIS
    }

# ============================================================
#  WEARABLE DEPLOYABILITY  (Section 4)
# ============================================================
def compute_deployability(energy_J: float) -> dict:
    """
    Energy-Based Deployability Estimation for wearable deployment.

    Battery capacity: WEARABLE_BATTERY_MAH mAh @ WEARABLE_BATTERY_V V
    = WEARABLE_BATTERY_J Joules usable (assumes 100% DoD for worst-case).

    Inferences per charge  = battery_J / energy_per_inference_J
    Runtime at target Hz   = inferences_per_charge / Hz  (seconds → hours)
    """
    if energy_J <= 0:
        return {"error": "energy <= 0, cannot estimate deployability"}

    inferences_per_charge = WEARABLE_BATTERY_J / energy_J
    runtime_s  = inferences_per_charge / INFERENCE_RATE_HZ
    runtime_hr = runtime_s / 3600.0

    return {
        "battery_capacity_J":      WEARABLE_BATTERY_J,
        "battery_mAh":             WEARABLE_BATTERY_MAH,
        "battery_V":               WEARABLE_BATTERY_V,
        "energy_per_inference_J":  energy_J,
        "inference_rate_Hz":       INFERENCE_RATE_HZ,
        "inferences_per_charge":   inferences_per_charge,
        "runtime_at_target_hz_hr": runtime_hr,
        "runtime_at_target_hz_min": runtime_hr * 60.0,
    }

def get_deterministic_input(batch_size, device):
    gen = torch.Generator(device=device)
    gen.manual_seed(42)
    return torch.randn(batch_size, 3, 224, 224,
                       device=device, dtype=torch.float16, generator=gen)

# ============================================================
#  SINGLE BENCHMARK PASS
# ============================================================
def run_benchmark(device='cuda', output_file='benchmark/results/benchmark_results.json'):
    BATCH_SIZE = 1
    results    = {}
    gpu_idx    = _gpu_index(device)

    print(f"Device: {device}  (GPU index for NVML: {gpu_idx})")
    print("-" * 60)

    # ------------------------------------------------------------------ CLIP
    force_cleanup()
    print("Benchmarking: CLIP Baseline")
    try:
        config = load_config_from_yaml("configs/clip_baseline.yaml")
        config.device = "cpu"
        model = get_model("clip", config)
        model = strict_fp16_setup(model, device)
        dummy = get_deterministic_input(BATCH_SIZE, device)
        vis_func = lambda: model.encode_image(dummy)

        stats, _  = measure_latency_stats(vis_func, device=device)
        mem       = measure_peak_memory(vis_func, device)
        pwr       = measure_power_and_energy(vis_func, gpu_idx, device=device)
        safety    = compute_safety_margin(stats)
        deploy    = compute_deployability(pwr["energy_J"])

        results["CLIP"] = {
            "latency": stats,
            "memory_MB": mem,
            "power_energy": pwr,
            "safety_margin": safety,
            "deployability": deploy,
        }
        print(f"  -> Mean: {stats['mean_ms']:.2f}ms | p99: {stats['p99_ms']:.2f}ms | "
              f"WCL: {safety['worst_case_latency_ms']:.2f}ms | "
              f"Power: {pwr['power_W']:.2f}W | Energy: {pwr['energy_mJ']:.3f}mJ | "
              f"RT OK: {safety['satisfies_rt_constraint']}")
        del model
    except Exception as e:
        print(f"  x CLIP Failed: {e}")

    # ------------------------------------------------------------------ LoRA
    force_cleanup()
    print("\nBenchmarking: LoRA")
    try:
        config = load_config_from_yaml("configs/clip_lora.yaml")
        config.device = "cpu"
        model = get_model("clip_lora", config)

        if hasattr(model, 'model') and os.path.exists(CHECKPOINT_PATHS["LORA_ADAPTER"]):
            model.model = PeftModel.from_pretrained(
                model.model.get_base_model(),
                CHECKPOINT_PATHS["LORA_ADAPTER"],
                is_trainable=True
            )

        model = strict_fp16_setup(model, device)
        dummy = get_deterministic_input(BATCH_SIZE, device)
        vis_func = lambda: model.encode_image(dummy)

        # Unmerged
        print("  -> Measuring Unmerged...")
        stats_un, _ = measure_latency_stats(vis_func, device=device)
        mem_un      = measure_peak_memory(vis_func, device)
        pwr_un      = measure_power_and_energy(vis_func, gpu_idx, device=device)
        safety_un   = compute_safety_margin(stats_un)
        deploy_un   = compute_deployability(pwr_un["energy_J"])

        # Merged
        print("  -> Merging...")
        if hasattr(model.model, 'merge_and_unload'):
            model.model = model.model.merge_and_unload()

        print("  -> Measuring Merged...")
        stats_mg, _ = measure_latency_stats(vis_func, device=device)
        mem_mg      = measure_peak_memory(vis_func, device)
        pwr_mg      = measure_power_and_energy(vis_func, gpu_idx, device=device)
        safety_mg   = compute_safety_margin(stats_mg)
        deploy_mg   = compute_deployability(pwr_mg["energy_J"])

        results["LoRA"] = {
            "unmerged": {
                "latency": stats_un, "memory_MB": mem_un,
                "power_energy": pwr_un, "safety_margin": safety_un,
                "deployability": deploy_un,
            },
            "merged": {
                "latency": stats_mg, "memory_MB": mem_mg,
                "power_energy": pwr_mg, "safety_margin": safety_mg,
                "deployability": deploy_mg,
            },
        }
        del model
    except Exception as e:
        print(f"  x LoRA Failed: {e}")

    # ---------------------------------------------------------------- Frozen
    force_cleanup()
    print("\nBenchmarking: Frozen")
    try:
        config = load_config_from_yaml("configs/frozen_clip.yaml")
        config.device = "cpu"
        model = get_model("frozen", config)

        if os.path.exists(CHECKPOINT_PATHS["FROZEN"]):
            ckpt = torch.load(CHECKPOINT_PATHS["FROZEN"], map_location='cpu')
            state_dict = ckpt.get('model_state', ckpt)
            new_sd = {
                k.replace('vision_encoder.', ''): v
                for k, v in state_dict.items()
                if k.startswith('vision_encoder.')
            }
            model.vision_encoder.load_state_dict(new_sd, strict=False)

        model = strict_fp16_setup(model, device)
        dummy = get_deterministic_input(BATCH_SIZE, device)

        # Vision-only (LLM offloaded)
        print("  -> Offloading LLM to CPU for strict vision measurement...")
        model.language_model.to("cpu")
        force_cleanup()

        vis_func    = lambda: model.encode_image(dummy)
        mem_vis     = measure_peak_memory(vis_func, device)
        stats_vis, _ = measure_latency_stats(vis_func, device=device)
        pwr_vis     = measure_power_and_energy(vis_func, gpu_idx, device=device)
        safety_vis  = compute_safety_margin(stats_vis)
        deploy_vis  = compute_deployability(pwr_vis["energy_J"])

        # E2E generation (LLM back on GPU)
        print("  -> Reloading LLM to GPU for E2E generation...")
        model.language_model.to(device)
        force_cleanup()

        def gen_func():
            with torch.no_grad():
                model.generate(dummy, model.tokenizer,
                               max_length=10, temperature=1.0, top_k=50)

        # Generation is slow (500ms–2s per call on GPT-2 Large).
        # 50 runs is statistically sufficient for p99 on a slow function —
        # with 50 samples, p99 = average of the worst sample, which is stable
        # enough given the low variance of autoregressive generation latency.
        # 300 runs here would take 5–10 minutes per benchmark pass.
        print("  -> Measuring E2E generation (50 runs, please wait ~1-2 min)...")
        stats_e2e, _ = measure_latency_stats(gen_func, runs=50, warmup=5, device=device, label="E2E-gen")
        print(f"     Latency done: mean={stats_e2e['mean_ms']:.0f}ms, p99={stats_e2e['p99_ms']:.0f}ms")
        print("  -> Measuring E2E peak memory...")
        mem_e2e      = measure_peak_memory(gen_func, device)
        print("  -> Measuring E2E power (20 runs)...")
        pwr_e2e      = measure_power_and_energy(gen_func, gpu_idx, runs=20, warmup=2, device=device)
        safety_e2e   = compute_safety_margin(stats_e2e)
        deploy_e2e   = compute_deployability(pwr_e2e["energy_J"])

        results["Frozen"] = {
            "vision": {
                "latency": stats_vis, "memory_MB": mem_vis,
                "power_energy": pwr_vis, "safety_margin": safety_vis,
                "deployability": deploy_vis,
            },
            "e2e": {
                "latency": stats_e2e, "memory_MB": mem_e2e,
                "power_energy": pwr_e2e, "safety_margin": safety_e2e,
                "deployability": deploy_e2e,
            },
        }
        del model
    except Exception as e:
        print(f"  x Frozen Failed: {e}")

    # ------------------------------------------------------------------ SAVE
    results["_meta"] = {
        "determinism_warnings": _determinism_warnings,
        "seed": 42,
        "batch_size": BATCH_SIZE,
        "latency_runs": 300,
        "latency_warmup_runs": 30,
        "power_sampling_interval_ms": 1.0,
        "os_jitter_ms": OS_JITTER_MS,
        "delay_buffer_ms": DELAY_BUFFER_MS,
        "wearable_battery_mAh": WEARABLE_BATTERY_MAH,
        "wearable_battery_V": WEARABLE_BATTERY_V,
        "target_inference_rate_Hz": INFERENCE_RATE_HZ,
    }

    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        print("-" * 60)
        print(f"[SUCCESS] Results saved to {output_path}")
    except Exception as e:
        print(f"Error saving: {e}")

    return results


# ============================================================
#  MAIN: run 3 times, save as delay_aware_results_{1,2,3}.json
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',      default='cuda')
    parser.add_argument('--output_dir',  default='benchmark/results')
    args = parser.parse_args()

    for i in range(1, 4):
        print(f"\n{'='*60}")
        print(f"  BENCHMARK RUN {i}/3")
        print(f"{'='*60}")
        set_seed(42)   # re-seed before every run for reproducibility
        output_file = f"{args.output_dir}/delay_aware_results_update_2_{i}.json"
        run_benchmark(device=args.device, output_file=output_file)

    print("\n[DONE] All 3 runs complete.")