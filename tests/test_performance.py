import os
import sys
import time

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    import pytest
except ImportError:
    pytest = None
from src.reasoning.verifier import ClinicalVerifier
from src.retrieval.hybrid_retriever import HybridClinicalRetriever
from src.agents.workflow import build_safecds_graph
from src.agents.llm_client import LLMClinicalAgent


def test_verifier_clinical_rules():
    ontology_path = os.path.join("ontologies", "cardiometabolic_core.owl")
    verifier = ClinicalVerifier(ontology_path=ontology_path)

    # 1. Metformin + CKD Stage 4/5 must fail
    res_met = verifier.verify_prescription("P1", ["CKD_Stage_4_5"], ["Metformin"])
    assert res_met["status"] == "FAIL"
    assert "Metformin is contraindicated" in res_met["violated_axiom"]

    # 2. Lisinopril + CKD Stage 4/5 must fail
    res_lis = verifier.verify_prescription("P2", ["CKD_Stage_4_5"], ["Lisinopril"])
    assert res_lis["status"] == "FAIL"
    assert "Lisinopril" in res_lis["violated_axiom"]

    # 3. Insulin + CKD Stage 4/5 must pass
    res_ins = verifier.verify_prescription("P3", ["CKD_Stage_4_5"], ["Insulin"])
    assert res_ins["status"] == "PASS"
    assert res_ins["violated_axiom"] is None

    # 4. Amlodipine + CKD Stage 4/5 must pass
    res_aml = verifier.verify_prescription("P4", ["CKD_Stage_4_5"], ["Amlodipine"])
    assert res_aml["status"] == "PASS"

    # 5. Metformin + Type 2 Diabetes without CKD must pass
    res_dm = verifier.verify_prescription("P5", ["Type_2_Diabetes"], ["Metformin"])
    assert res_dm["status"] == "PASS"


def test_verifier_caching_performance():
    ontology_path = os.path.join("ontologies", "cardiometabolic_core.owl")
    verifier = ClinicalVerifier(ontology_path=ontology_path, enable_cache=True)
    verifier.clear_cache()

    # First call (cache miss)
    t0 = time.perf_counter()
    res1 = verifier.verify_prescription("P1", ["CKD_Stage_4_5", "Type_2_Diabetes"], ["Metformin"])
    t_first = time.perf_counter() - t0

    assert verifier.cache_misses == 1
    assert verifier.cache_hits == 0

    # Repeat 100 times (cache hits)
    t0 = time.perf_counter()
    for _ in range(100):
        res2 = verifier.verify_prescription("P_different_id", ["CKD_Stage_4_5", "Type_2_Diabetes"], ["Metformin"])
        assert res2["status"] == res1["status"]
    t_cached = (time.perf_counter() - t0) / 100

    assert verifier.cache_hits == 100
    assert verifier.cache_stats["hit_ratio"] > 0.98
    # Cached access should be sub-millisecond
    assert t_cached < 0.001


def test_hybrid_retriever_caching():
    retriever = HybridClinicalRetriever()
    query = "Management of patient with CKD_Stage_4_5"

    # First query
    ctx1 = retriever.retrieve_context(query, top_k=2)
    assert len(ctx1) > 0
    assert "KDIGO" in ctx1 or "ADA" in ctx1 or "ACC" in ctx1

    # Second query (cached)
    t0 = time.perf_counter()
    ctx2 = retriever.retrieve_context(query, top_k=2)
    t_cached = time.perf_counter() - t0

    assert ctx1 == ctx2
    assert t_cached < 0.001  # Instant retrieval via LRU cache


def test_workflow_end_to_end_self_correction():
    ontology_path = os.path.join("ontologies", "cardiometabolic_core.owl")
    verifier = ClinicalVerifier(ontology_path=ontology_path)
    llm_agent = LLMClinicalAgent()
    retriever = HybridClinicalRetriever()

    app = build_safecds_graph(verifier=verifier, llm_agent=llm_agent, retriever=retriever)

    # Test patient presenting with CKD Stage 4/5 and Hypertension
    patient_input = {
        "patient_id": "PT_TEST_CORRECT",
        "conditions": ["CKD_Stage_4_5", "Hypertension"],
        "retrieved_context": None,
        "proposed_medication": None,
        "reasoning": None,
        "retries": 0,
        "max_retries": 3,
        "violations": [],
        "status": "PENDING",
        "final_output": {}
    }

    result = app.invoke(patient_input)
    final_output = result["final_output"]

    assert final_output["status"] == "APPROVED"
    assert "Amlodipine" in final_output["recommended_treatment"]
    # Verified that self-correction caught Lisinopril contraindication in audit trail
    assert len(final_output["audit_trail"]) >= 1
    assert any("Lisinopril" in v for v in final_output["audit_trail"])


if __name__ == "__main__":
    test_verifier_clinical_rules()
    test_verifier_caching_performance()
    test_hybrid_retriever_caching()
    test_workflow_end_to_end_self_correction()
    print("All performance and verification tests passed successfully!")
