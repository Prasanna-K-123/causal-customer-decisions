"""Portable stage timings and sampled RSS; no production-performance claim."""
from __future__ import annotations
import contextlib, hashlib, json, os, platform, sys, threading, time
from pathlib import Path
import importlib.metadata
import psutil

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda:f.read(1<<20), b""):
            h.update(b)
    return h.hexdigest()

def dump_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temp.replace(path)

class Measurement:
    def __init__(self, output):
        self.output, self.stages, self.peak = Path(output), [], 0
        self.start = time.perf_counter()
        self.memory_error = None
        try:
            self.process = psutil.Process()
        except (OSError, psutil.Error) as exc:
            self.process = None
            self.memory_error = f"{type(exc).__name__}: {exc}"

    def sample_rss(self):
        """Timing still works when a sandbox does not expose process memory."""
        if self.process is None:
            return None
        try:
            return self.process.memory_info().rss
        except (OSError, psutil.Error) as exc:
            self.memory_error = f"{type(exc).__name__}: {exc}"
            return None
    @contextlib.contextmanager
    def stage(self, name):
        stop = threading.Event()
        peaks = []
        def sample():
            value = self.sample_rss()
            if value is not None:
                peaks.append(value)
        sample()
        def monitor():
            while not stop.wait(.05):
                sample()
        worker = threading.Thread(target=monitor, daemon=True)
        w, c = time.perf_counter(), time.process_time()
        worker.start()
        print(json.dumps({"stage":name,"status":"started"}), flush=True)
        try:
            yield
        finally:
            stop.set(); worker.join()
            sample()
            record = {"stage":name,"wall_seconds":time.perf_counter()-w,
                      "process_cpu_seconds":time.process_time()-c,
                      "observed_peak_rss_bytes":max(peaks) if peaks else None,
                      "rss_sample_count":len(peaks),
                      "rss_measurement_error":self.memory_error}
            if peaks:
                self.peak = max(self.peak,max(peaks))
            self.stages.append(record)
            self.save()
            print(json.dumps(record), flush=True)
    def save(self):
        versions = {k:importlib.metadata.version(k) for k in
                    ["numpy","pandas","scipy","scikit-learn","matplotlib","psutil"]}
        dump_json(self.output,{"stages":self.stages,"wall_since_start_seconds":time.perf_counter()-self.start,
            "observed_process_peak_rss_bytes":self.peak or None,"sampling_interval_seconds":.05,
            "rss_measurement_status":"unavailable_or_partial" if self.memory_error else "sampled",
            "rss_measurement_error":self.memory_error,
            "measurement_limit":"Sampled RSS; excludes sibling processes; local shared-machine timings",
            "environment":{"python":sys.version,"platform":platform.platform(),
                "processor":platform.processor(),"logical_cpus":os.cpu_count(),"versions":versions,
                "thread_environment":{k:os.environ.get(k) for k in
                    ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]}}})
