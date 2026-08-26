from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from src.agents.llm_client import LLMClinicalAgent

class AgentState(TypedDict):
    patient_id: str
    conditions: List[str]
    proposed_medication: Optional[str]
    reasoning: Optional[str]
    retries: int
    max_retries: int
    violations: List[str]
    status: str
    final_output: dict

# Initialize LLM Agent instance
llm_agent = LLMClinicalAgent()

def generate_hypothesis_node(state: AgentState):
    """
    Invokes LLaMA via Ollama dynamically with the accumulated violation feedback.
    """
    attempt = state["retries"] + 1
    print(f"\n[LLaMA Agent] Generating hypothesis (Attempt {attempt})...")
    
    response = llm_agent.generate_recommendation(
        patient_id=state["patient_id"],
        conditions=state["conditions"],
        violations=state["violations"]
    )
    
    state["proposed_medication"] = response["medication"]
    state["reasoning"] = response["reasoning"]
    print(f"[LLaMA Agent] Proposed Medication: {state['proposed_medication']}")
    print(f"[LLaMA Agent] Clinical Rationale: {state['reasoning']}")
    
    return state

def verify_node(state: AgentState, verifier):
    """
    Neuro-symbolic verification gate using HermiT Description Logic.
    """
    print(f"\n[HermiT Gate] Verifying '{state['proposed_medication']}' against ontology...")
    
    result = verifier.verify_prescription(
        patient_id=state["patient_id"],
        conditions=state["conditions"],
        medications=[state["proposed_medication"]]
    )
    
    if result["status"] == "PASS":
        print("[HermiT Gate] RESULT: PASSED (Zero Ontological Violations).")
        state["status"] = "VERIFIED"
    else:
        print(f"[HermiT Gate] RESULT: FAILED -> {result['violated_axiom']}")
        state["status"] = "FAILED"
        state["violations"].append(result["violated_axiom"])
        state["retries"] += 1
        
    return state

def route_next_step(state: AgentState):
    if state["status"] == "VERIFIED":
        return "finalize"
    elif state["retries"] >= state["max_retries"]:
        return "escalate"
    else:
        return "retry"

def finalize_node(state: AgentState):
    state["final_output"] = {
        "status": "APPROVED",
        "patient_id": state["patient_id"],
        "recommended_treatment": state["proposed_medication"],
        "clinical_rationale": state["reasoning"],
        "total_attempts": state["retries"] + 1,
        "audit_trail": state["violations"]
    }
    return state

def escalate_node(state: AgentState):
    state["final_output"] = {
        "status": "ESCALATED_FOR_HUMAN_REVIEW",
        "patient_id": state["patient_id"],
        "last_proposed_treatment": state["proposed_medication"],
        "reason": "Maximum retry limit reached without resolving formal clinical violations.",
        "audit_trail": state["violations"]
    }
    return state

def build_safecds_graph(verifier):
    workflow = StateGraph(AgentState)
    
    workflow.add_node("generate_hypothesis", generate_hypothesis_node)
    workflow.add_node("verify", lambda state: verify_node(state, verifier))
    workflow.add_node("finalize", finalize_node)
    workflow.add_node("escalate", escalate_node)
    
    workflow.set_entry_point("generate_hypothesis")
    workflow.add_edge("generate_hypothesis", "verify")
    
    workflow.add_conditional_edges(
        "verify",
        route_next_step,
        {
            "finalize": "finalize",
            "retry": "generate_hypothesis",
            "escalate": "escalate"
        }
    )
    
    workflow.add_edge("finalize", END)
    workflow.add_edge("escalate", END)
    
    return workflow.compile()