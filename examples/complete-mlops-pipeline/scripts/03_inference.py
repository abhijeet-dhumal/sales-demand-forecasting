#!/usr/bin/env python3
"""
KServe Inference Testing
Usage: python 03_inference.py
"""
import os, sys, time
import requests, numpy as np
import urllib3; urllib3.disable_warnings()

NAMESPACE = os.getenv("NAMESPACE", "feast-trainer-demo")
INFERENCE_URL = os.getenv("INFERENCE_URL", f"https://sales-forecast-{NAMESPACE}.apps.oai-kft-ibm.ibm.rh-ods.com")
TOKEN = os.getenv("KSERVE_TOKEN") or os.getenv("K8S_TOKEN")


class Client:
    def __init__(self, url, token=None):
        self.url = url.rstrip("/")
        self.s = requests.Session()
        self.s.verify = False
        if token: self.s.headers["Authorization"] = f"Bearer {token}"
    
    def health(self): return self.s.get(f"{self.url}/health", timeout=10).json()
    def info(self): return self.s.get(f"{self.url}/v1/models/sales-forecast", timeout=10).json()
    def predict(self, instances): return self.s.post(f"{self.url}/v1/models/sales-forecast:predict", json={"instances": instances}, timeout=30).json()
    def explain(self): return self.s.post(f"{self.url}/v1/models/sales-forecast:explain", timeout=10).json()


if __name__ == "__main__":
    # Auto-discover from cluster if possible
    url = INFERENCE_URL
    try:
        from kubernetes import client, config
        config.load_kube_config()
        api = client.CustomObjectsApi()
        isvc = api.get_namespaced_custom_object("serving.kserve.io", "v1beta1", NAMESPACE, "inferenceservices", "sales-forecast")
        url = isvc.get("status", {}).get("url") or url
    except: pass

    print(f"{'='*60}\nKServe Inference Testing\nURL: {url}\n{'='*60}")
    client = Client(url, TOKEN)
    passed, total = 0, 5

    # Health
    print("\n🏥 Health...")
    try: h = client.health(); print(f"   ✅ {h.get('status')}"); passed += 1
    except Exception as e: print(f"   ❌ {e}")

    # Info
    print("\n📋 Model Info...")
    try: info = client.info(); print(f"   ✅ {info.get('name')} ({info.get('input_dim')} features)"); passed += 1
    except Exception as e: print(f"   ❌ {e}")

    # Single predict
    print("\n📊 Single Prediction...")
    sample = [25000, 24000, 23000, 22000, 20000, 24500, 150000, 65.0, 2.8, 215.0, 5.5]
    try: pred = client.predict([sample])["predictions"][0]; print(f"   ✅ ${pred:,.2f}"); passed += 1
    except Exception as e: print(f"   ❌ {e}")

    # Batch
    print("\n📊 Batch Prediction...")
    batch = [sample, [50000,48000,45000,42000,40000,47000,250000,70,3,220,4], [15000,14500,14000,13500,13000,14200,80000,55,2.5,210,6.5]]
    try:
        preds = client.predict(batch)["predictions"]
        for i, p in enumerate(preds): print(f"   {i+1}: ${p:,.2f}")
        passed += 1
    except Exception as e: print(f"   ❌ {e}")

    # Latency
    print("\n⏱️  Latency (10 req)...")
    try:
        lats = [(time.time(), client.predict([sample]), time.time()) for _ in range(10)]
        lats = [(l[2]-l[0])*1000 for l in lats]
        print(f"   Mean: {np.mean(lats):.0f}ms | P95: {np.percentile(lats,95):.0f}ms")
        passed += 1
    except Exception as e: print(f"   ❌ {e}")

    print(f"\n{'='*60}\n{'✅' if passed==total else '⚠️'} Tests: {passed}/{total}\n{'='*60}")
    sys.exit(0 if passed == total else 1)
