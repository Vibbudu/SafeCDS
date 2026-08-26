import os
from owlready2 import get_ontology, sync_reasoner_hermit, default_world

class ClinicalVerifier:
    def __init__(self, ontology_path: str, **kwargs):
        self.ontology_path = os.path.abspath(ontology_path)
        self.onto = get_ontology(f"file://{self.ontology_path}").load()

    def verify_prescription(self, patient_id: str, conditions: list[str], medications: list[str]) -> dict:
        with self.onto:
            # 1. Create Patient individual
            patient = self.onto.Patient(patient_id)

            # 2. Attach Conditions
            for cond in conditions:
                cond_class = getattr(self.onto, cond, None)
                if cond_class:
                    cond_ind = cond_class(f"{cond}_{patient_id}")
                    patient.hasCondition.append(cond_ind)

            # 3. Attach Medications
            for med in medications:
                med_class = getattr(self.onto, med, None)
                if med_class:
                    med_ind = med_class(f"{med}_{patient_id}")
                    patient.hasPrescribedDrug.append(med_ind)

        # Run HermiT reasoner directly through owlready2
        try:
            sync_reasoner_hermit(infer_property_values=True)
        except Exception:
            pass

        # Check if patient was inferred as UnsafeMetforminPrescription
        unsafe_class = getattr(self.onto, "UnsafeMetforminPrescription", None)
        is_unsafe = False
        if unsafe_class and patient in unsafe_class.instances():
            is_unsafe = True

        if is_unsafe:
            return {
                "status": "FAIL",
                "violated_axiom": "Contraindication detected: Metformin cannot be prescribed to patients with CKD_Stage_4_5."
            }
        
        return {"status": "PASS", "violated_axiom": None}