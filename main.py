import os
import json
from src.reasoning.verifier import ClinicalVerifier
from src.agents.workflow import build_safecds_graph

ONTOLOGY_PATH = os.path.join("ontologies", "cardiometabolic_core.owl")

print("Initializing SafeCDS Reasoning Engine...")
verifier = ClinicalVerifier(ontology_path=ONTOLOGY_PATH)

print("Building LangGraph Orchestration State Machine...")
safecds_app = build_safecds_graph(verifier)

# Test Patient: Patient presenting with CKD Stage 4/5 (where standard Type-2 diabetes first-line Metformin is unsafe)
patient_input = {
    "patient_id": "Patient_Cardio_102",
    "conditions": ["CKD_Stage_4_5", "Type_2_Diabetes"],
    "proposed_medication": None,
    "reasoning": None,
    "retries": 0,
    "max_retries": 3,
    "violations": [],
    "status": "PENDING",
    "final_output": {}
}

print("\nExecuting SafeCDS Pipeline with Dynamic LLaMA-3.2 Reasoning...")
result = safecds_app.invoke(patient_input)

print("\n" + "=" * 45)
print("             FINAL DECISION AUDIT            ")
print("=" * 45)
print(json.dumps(result["final_output"], indent=4))