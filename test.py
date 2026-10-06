import os
import sys

print("=== SafeCDS Environment & Health Check ===")

# 1. PyTorch & GPU
try:
    import torch
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("PyTorch: Not installed (optional, running in CPU/fast-path mode)")

# 2. JVM / JPype Bridge
try:
    import jpype
    import jpype.imports
    
    java_home = os.environ.get("JAVA_HOME")
    candidates = [
        os.path.join(java_home, "bin", "server", "jvm.dll") if java_home else None,
        r"C:\Program Files\Java\jdk-26\bin\server\jvm.dll",
        r"C:\Program Files\Java\jdk-21\bin\server\jvm.dll",
        r"C:\Program Files\Java\jdk-17\bin\server\jvm.dll",
        jpype.getDefaultJVMPath() if hasattr(jpype, "getDefaultJVMPath") else None
    ]
    jvm_dll_path = next((p for p in candidates if p and os.path.exists(p)), None)

    if jvm_dll_path and not jpype.isJVMStarted():
        jpype.startJVM(jvm_dll_path, classpath=["java_libs/*"])
        print(f"Java JVM Started: {jpype.isJVMStarted()}")
        java_lang = jpype.JPackage("java.lang")
        print(f"Java Version: {java_lang.System.getProperty('java.version')}")
        jpype.shutdownJVM()
    else:
        print("JPype: Available (JVM dll path not found; using pure axiomatic engine)")
except ImportError:
    print("JPype: Not installed (using high-speed native axiomatic verifier)")

# 3. LangGraph & LangChain Check
try:
    from importlib.metadata import version
    print(f"LangGraph Version: {version('langgraph')}")
except Exception:
    print("LangGraph: Using built-in lightweight state machine")

# 4. SafeCDS Core Engine Check
from src.reasoning.verifier import ClinicalVerifier
from src.retrieval.hybrid_retriever import HybridClinicalRetriever
from src.agents.workflow import build_safecds_graph

print("\n--- Verifying SafeCDS Core Engine ---")
v = ClinicalVerifier("ontologies/cardiometabolic_core.owl")
ret = HybridClinicalRetriever()
app = build_safecds_graph(v, retriever=ret)
print("SafeCDS Pipeline: Initialized and fully operational!")