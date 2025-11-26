import os
import logging

LIGHTEN = int(os.environ.get("DEEPDOC_LIGHTEN", "0"))

PARALLEL_DEVICES = 0
try:
    import torch.cuda
    PARALLEL_DEVICES = torch.cuda.device_count()
    logging.info(f"Found {PARALLEL_DEVICES} GPUs")
except Exception as e:
    logging.info("Can't import package 'torch' or access GPU: %s", str(e))

# CPU 并行处理支持
# 当没有 GPU 时，可以启用 CPU 线程池并行处理
if PARALLEL_DEVICES == 0:
    use_original = os.environ.get("DEEPDOC_USE_ORIGINAL", "0").lower() in ("1", "true", "yes")
    
    if use_original:
        PARALLEL_DEVICES = 0  # 显式设置为 0，保持原始行为
        print(f"[SETTINGS DEBUG] Using ORIGINAL deepdoc behavior (no CPU parallelism)")
        logging.info("Using original deepdoc behavior (no CPU parallelism)")
    else:
        # 自动检测 CPU 核心数并预留部分核心
        cpu_count = os.cpu_count() or 1
        reserve_cores = int(os.environ.get("DEEPDOC_RESERVE_CPU", "4"))
        available_cores = max(1, cpu_count - reserve_cores)
        
        # 手动指定并行线程数（优先级最高）
        manual_override = os.environ.get("DEEPDOC_PARALLEL_THREADS")
        if manual_override:
            PARALLEL_DEVICES = int(manual_override)
            print(f"[SETTINGS DEBUG] Using manual override: PARALLEL_DEVICES={PARALLEL_DEVICES}")
        else:
            PARALLEL_DEVICES = available_cores
            print(f"[SETTINGS DEBUG] Auto-detected CPU cores: {cpu_count}, reserved: {reserve_cores}, using: {PARALLEL_DEVICES}")
        
        logging.info(f"Using CPU thread pool parallelism with {PARALLEL_DEVICES} threads (total cores: {cpu_count}, reserved: {reserve_cores})")