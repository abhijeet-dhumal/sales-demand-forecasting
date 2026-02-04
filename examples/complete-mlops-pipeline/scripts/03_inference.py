#!/usr/bin/env python3
"""
KServe Model Deployment + Inference Testing
Usage: python3.12 03_inference.py
"""
import os, sys, time, urllib3
from pathlib import Path
from datetime import datetime
import requests, numpy as np
urllib3.disable_warnings()

# Config
NAMESPACE = os.getenv("NAMESPACE", "feast-trainer-demo")
MODEL_NAME = os.getenv("MODEL_NAME", "sales-forecast")
SHARED_PVC = os.getenv("SHARED_PVC", "feast-pvc")
K8S_TOKEN = os.getenv("K8S_TOKEN")
K8S_API_SERVER = os.getenv("K8S_API_SERVER")


class InferenceClient:
    """Simple client for testing the deployed model"""
    def __init__(self, url, token=None):
        self.url = url.rstrip("/")
        self.s = requests.Session()
        self.s.verify = False
        if token: self.s.headers["Authorization"] = f"Bearer {token}"
    
    def health(self): return self.s.get(f"{self.url}/health", timeout=10).json()
    def info(self): return self.s.get(f"{self.url}/v1/models/{MODEL_NAME}", timeout=10).json()
    def predict(self, instances): return self.s.post(f"{self.url}/v1/models/{MODEL_NAME}:predict", json={"instances": instances}, timeout=30).json()


def deploy():
    """Deploy model to KServe"""
    from kubernetes import client as k8s, config
    from kserve import KServeClient, V1beta1InferenceService, V1beta1InferenceServiceSpec, V1beta1PredictorSpec
    
    deploy_id = datetime.now().strftime("%m%d-%H%M")
    labels = {"app": "sales-forecasting", "deploy-id": deploy_id}
    
    print(f"{'='*60}\n🚀 KServe Deployment\nNamespace: {NAMESPACE} | Deploy ID: {deploy_id}\n{'='*60}")

    # Auth
    if K8S_TOKEN and K8S_API_SERVER:
        cfg = k8s.Configuration()
        cfg.host, cfg.verify_ssl = K8S_API_SERVER, False
        cfg.api_key = {"authorization": f"Bearer {K8S_TOKEN}"}
        k8s.Configuration.set_default(cfg)
    else:
        config.load_kube_config()
    
    core, custom, kserve = k8s.CoreV1Api(), k8s.CustomObjectsApi(), KServeClient()
    
    # 1. ConfigMap with serve script
    print("📦 ConfigMap...")
    serve_script = Path(__file__).parent / "serve.py"
    cm_name = f"{MODEL_NAME}-serve"
    try: core.delete_namespaced_config_map(cm_name, NAMESPACE)
    except: pass
    core.create_namespaced_config_map(NAMESPACE, k8s.V1ConfigMap(
        metadata=k8s.V1ObjectMeta(name=cm_name, labels=labels), 
        data={"serve.py": serve_script.read_text()}))
    
    # 2. InferenceService
    print("🚀 InferenceService...")
    try: 
        kserve.delete(MODEL_NAME, namespace=NAMESPACE)
        # Wait for deletion to complete
        for _ in range(30):
            try: kserve.get(MODEL_NAME, namespace=NAMESPACE); time.sleep(2)
            except: break
    except: pass
    
    isvc = V1beta1InferenceService(
        api_version="serving.kserve.io/v1beta1", kind="InferenceService",
        metadata=k8s.V1ObjectMeta(name=MODEL_NAME, namespace=NAMESPACE, labels=labels),
        spec=V1beta1InferenceServiceSpec(predictor=V1beta1PredictorSpec(
            containers=[k8s.V1Container(
                name="kserve-container", image="quay.io/modh/ray:2.52.1-py312-cu128",
                command=["/bin/bash", "-c", "pip install -q flask torch joblib numpy scikit-learn && python /scripts/serve.py"],
                env=[k8s.V1EnvVar(name="MODEL_DIR", value="/mnt/models")],
                ports=[k8s.V1ContainerPort(container_port=8080, protocol="TCP")],
                volume_mounts=[k8s.V1VolumeMount(name="model-storage", mount_path="/mnt/models", sub_path="models"),
                               k8s.V1VolumeMount(name="serve-script", mount_path="/scripts")],
                resources=k8s.V1ResourceRequirements(limits={"cpu": "2", "memory": "4Gi"}, requests={"cpu": "1", "memory": "2Gi"})
            )],
            volumes=[k8s.V1Volume(name="model-storage", persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(claim_name=SHARED_PVC)),
                     k8s.V1Volume(name="serve-script", config_map=k8s.V1ConfigMapVolumeSource(name=cm_name))])))
    
    kserve.create(isvc, namespace=NAMESPACE)
    
    # 3. Wait
    print("⏳ Waiting...")
    kserve.wait_isvc_ready(MODEL_NAME, namespace=NAMESPACE, timeout_seconds=300)
    internal_url = kserve.get(MODEL_NAME, namespace=NAMESPACE).get("status", {}).get("url", "")
    print(f"✅ Internal: {internal_url}")
    
    # 4. Route
    print("🌐 Route...")
    route = {"apiVersion": "route.openshift.io/v1", "kind": "Route",
             "metadata": {"name": MODEL_NAME, "namespace": NAMESPACE, "labels": labels},
             "spec": {"to": {"kind": "Service", "name": f"{MODEL_NAME}-predictor", "weight": 100},
                      "port": {"targetPort": "http"}, "tls": {"termination": "edge", "insecureEdgeTerminationPolicy": "Redirect"}}}
    try: custom.delete_namespaced_custom_object("route.openshift.io", "v1", NAMESPACE, "routes", MODEL_NAME)
    except: pass
    custom.create_namespaced_custom_object("route.openshift.io", "v1", NAMESPACE, "routes", route)
    route_host = custom.get_namespaced_custom_object("route.openshift.io", "v1", NAMESPACE, "routes", MODEL_NAME).get("spec", {}).get("host", "")
    external_url = f"https://{route_host}"
    print(f"✅ External: {external_url}")
    
    return external_url


def test(url):
    """Test the deployed model"""
    print(f"\n{'='*60}\n🧪 Inference Testing\nURL: {url}\n{'='*60}")
    client = InferenceClient(url, K8S_TOKEN)
    passed = 0
    
    # Health
    print("\n🏥 Health...")
    try: print(f"   ✅ {client.health().get('status')}"); passed += 1
    except Exception as e: print(f"   ❌ {e}")
    
    # Info
    print("\n📋 Model Info...")
    try: info = client.info(); print(f"   ✅ {info.get('name')} ({info.get('input_dim')} features)"); passed += 1
    except Exception as e: print(f"   ❌ {e}")
    
    # Predict with named features
    print("\n📊 Predictions...")
    # Named features are much clearer than raw arrays!
    sample = {
        "lag_1": 25000,      # Last week sales
        "lag_2": 24000,      # 2 weeks ago
        "lag_4": 23000,      # 4 weeks ago
        "lag_8": 22000,      # 8 weeks ago
        "lag_52": 20000,     # Same week last year
        "rolling_mean_4w": 24500,  # 4-week rolling average
        "store_size": 150000,      # Store square footage
        "temperature": 65.0,       # Temperature (F)
        "fuel_price": 2.8,         # Fuel price ($)
        "cpi": 215.0,              # Consumer Price Index
        "unemployment": 5.5        # Unemployment rate (%)
    }
    try:
        pred = client.predict([sample])["predictions"][0]
        print(f"   Medium store, stable sales: ${pred:,.2f}")
        
        # More scenarios with meaningful descriptions
        high_performer = {"lag_1": 50000, "lag_2": 48000, "lag_4": 45000, "lag_8": 42000, "lag_52": 40000,
                         "rolling_mean_4w": 47000, "store_size": 250000, "temperature": 70, "fuel_price": 3, "cpi": 220, "unemployment": 4}
        small_store = {"lag_1": 15000, "lag_2": 14500, "lag_4": 14000, "lag_8": 13500, "lag_52": 13000,
                      "rolling_mean_4w": 14200, "store_size": 80000, "temperature": 55, "fuel_price": 2.5, "cpi": 210, "unemployment": 6.5}
        
        preds = client.predict([high_performer, small_store])["predictions"]
        print(f"   Large store, high sales: ${preds[0]:,.2f}")
        print(f"   Small store, low sales: ${preds[1]:,.2f}")
        passed += 1
    except Exception as e: print(f"   ❌ {e}")
    
    # Latency
    print("\n⏱️  Latency (10 requests)...")
    try:
        times = []
        for _ in range(10):
            t0 = time.time()
            client.predict([sample])
            times.append((time.time() - t0) * 1000)
        print(f"   Mean: {np.mean(times):.0f}ms | P95: {np.percentile(times, 95):.0f}ms")
        passed += 1
    except Exception as e: print(f"   ❌ {e}")
    
    print(f"\n{'='*60}\n{'✅' if passed==4 else '⚠️'} Tests: {passed}/4\n{'='*60}")
    return passed == 4


if __name__ == "__main__":
    url = deploy()
    time.sleep(5)  # Give pod time to start serving
    success = test(url)
    sys.exit(0 if success else 1)
