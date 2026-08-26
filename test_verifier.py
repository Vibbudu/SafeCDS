import os
from src.reasoning.verifier import ClinicalVerifier

ONTOLOGY_PATH = os.path.join("ontologies", "cardiometabolic_core.owl")
verifier = ClinicalVerifier(ontology_path=ONTOLOGY_PATH)

print("\n--- Test 1: Unsafe Case (Metformin + CKD Stage 4/5) ---")
result_unsafe = verifier.verify_prescription(
    patient_id="Patient_Unsafe",
    conditions=["CKD_Stage_4_5"],
    medications=["Metformin"]
)
print(f"Status: {result_unsafe['status']}")
print(f"Details: {result_unsafe['violated_axiom']}")

print("\n--- Test 2: Safe Case (Insulin + CKD Stage 4/5) ---")
result_safe = verifier.verify_prescription(
    patient_id="Patient_Safe",
    conditions=["CKD_Stage_4_5"],
    medications=["Insulin"]
)
print(f"Status: {result_safe['status']}")
print(f"Details: {result_safe['violated_axiom']}")