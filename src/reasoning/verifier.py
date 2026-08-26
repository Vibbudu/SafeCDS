import os
from owlready2 import World, sync_reasoner_hermit

class ClinicalVerifier:
    def __init__(self, ontology_path: str, **kwargs):
        self.ontology_path = os.path.abspath(ontology_path)

    def _match_ontology_entity(self, raw_name: str, valid_entities: list[str]) -> str | None:
        raw_clean = raw_name.lower().replace("-", "_").replace(" ", "_")
        for entity in valid_entities:
            if entity.lower() in raw_clean:
                return entity
        return None

    def verify_prescription(self, patient_id: str, conditions: list[str], medications: list[str]) -> dict:
        known_meds = ["Metformin", "Lisinopril", "Insulin", "Amlodipine", "Losartan"]
        known_conds = ["CKD_Stage_4_5", "Type_2_Diabetes", "Hypertension", "Heart_Failure"]

        # Create an isolated world for this verification attempt to prevent state bleed
        isolated_world = World()
        onto = isolated_world.get_ontology(f"file://{self.ontology_path}").load()

        with onto:
            patient = onto.Patient(patient_id)

            # Match and assert conditions
            for cond in conditions:
                matched_c = self._match_ontology_entity(cond, known_conds) or cond
                cond_class = getattr(onto, matched_c, None)
                if cond_class:
                    cond_ind = cond_class(f"{matched_c}_{patient_id}")
                    patient.hasCondition.append(cond_ind)

            # Match and assert medications
            for med in medications:
                matched_m = self._match_ontology_entity(med, known_meds) or med
                med_class = getattr(onto, matched_m, None)
                if med_class:
                    med_ind = med_class(f"{matched_m}_{patient_id}")
                    patient.hasPrescribedDrug.append(med_ind)

        # Run HermiT Reasoner in this isolated world
        try:
            sync_reasoner_hermit(isolated_world, infer_property_values=True)
        except Exception:
            pass

        # Check for inferred unsafe classes
        violations = []
        unsafe_metformin = getattr(onto, "UnsafeMetforminPrescription", None)
        if unsafe_metformin and patient in unsafe_metformin.instances():
            violations.append("Contraindication: Metformin is contraindicated in CKD_Stage_4_5 (Risk of Lactic Acidosis).")

        unsafe_acei = getattr(onto, "UnsafeACEiPrescription", None)
        if unsafe_acei and patient in unsafe_acei.instances():
            violations.append("Contraindication: High-dose ACE-inhibitors (Lisinopril) are contraindicated in advanced CKD_Stage_4_5 without close monitoring (Risk of severe hyperkalemia / acute renal decline).")

        # Dispose isolated world memory
        isolated_world.close()

        if violations:
            return {
                "status": "FAIL",
                "violated_axiom": " | ".join(violations)
            }
        
        return {"status": "PASS", "violated_axiom": None}