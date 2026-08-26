from importlib.metadata import version
import os
import torch
import jpype
import jpype.imports

print("=== SafeCDS Environment Check ===")

# 1. PyTorch & GPU
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

# 2. JVM / JPype Bridge
jvm_dll_path = r"C:\Program Files\Java\jdk-26\bin\server\jvm.dll"

if not jpype.isJVMStarted():
    jpype.startJVM(jvm_dll_path, classpath=["java_libs/*"])
    print(f"Java JVM Started: {jpype.isJVMStarted()}")
    
    java_lang = jpype.JPackage("java.lang")
    print(f"Java Version: {java_lang.System.getProperty('java.version')}")
    jpype.shutdownJVM()

# 3. LangGraph Version Check via metadata
langgraph_version = version("langgraph")
print(f"LangGraph Version: {langgraph_version}")

print("All core dependencies loaded successfully!")