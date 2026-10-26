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
        self.process = psutil.Process()
    @contextlib.contextmanager
    def stage(self, name):
        stop = threading.Event()
        peaks = [self.process.memory_info().rss]
        def monitor():
            while not stop.wait(.05):
                peaks.append(self.process.memory_info().rss)
        worker = threading.Thread(target=monitor, daemon=True)
        w, c = time.perf_counter(), time.process_time()
        worker.start()
        print(json.dumps({"stage":name,"status":"started"}), flush=True)
        try:
            yield
        finally:
            stop.set(); worker.join()
            peaks.append(self.process.memory_info().rss)
            record = {"stage":name,"wall_seconds":time.perf_counter()-w,
                      "process_cpu_seconds":time.process_time()-c,
                      "observed_peak_rss_bytes":max(peaks)}
            self.peak = max(self.peak,max(peaks))
            self.stages.append(record)
            self.save()
            print(json.dumps(record), flush=True)
    def save(self):
        versions = {k:importlib.metadata.version(k) for k in
                    ["numpy","pandas","scipy","scikit-learn","matplotlib","psutil"]}
        dump_json(self.output,{"stages":self.stages,"wall_since_start_seconds":time.perf_counter()-self.start,
            "observed_process_peak_rss_bytes":self.peak,"sampling_interval_seconds":.05,
            "measurement_limit":"Sampled RSS; excludes sibling processes; local shared-machine timings",
            "environment":{"python":sys.version,"platform":platform.platform(),
                "processor":platform.processor(),"logical_cpus":os.cpu_count(),"versions":versions,
                "thread_environment":{k:os.environ.get(k) for k in
                    ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]}}})
