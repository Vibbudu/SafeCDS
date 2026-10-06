from typing import TypedDict, List, Optional, Dict, Any, Callable
from src.agents.llm_client import LLMClinicalAgent
from src.retrieval.hybrid_retriever import HybridClinicalRetriever

# Import LangGraph if available, otherwise provide an exact lightweight state machine
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    END = "__END__"
    
    class StateGraph:
        """Lightweight native state machine matching LangGraph's API for zero-dependency execution."""
        def __init__(self, state_schema):
            self.state_schema = state_schema
            self.nodes: Dict[str, Callable] = {}
            self.edges: Dict[str, str] = {}
            self.conditional_edges: Dict[str, Tuple[Callable, Dict[str, str]]] = {}
            self.entry_point: Optional[str] = None

        def add_node(self, name: str, action: Callable):
            self.nodes[name] = action

        def set_entry_point(self, name: str):
            self.entry_point = name

        def add_edge(self, from_node: str, to_node: str):
            self.edges[from_node] = to_node

        def add_conditional_edges(self, from_node: str, condition: Callable, routing_map: Dict[str, str]):
            self.conditional_edges[from_node] = (condition, routing_map)

        def compile(self):
            return CompiledApp(self)

    class CompiledApp:
        def __init__(self, graph: StateGraph):
            self.graph = graph

        def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
            current_node = self.graph.entry_point
            current_state = dict(state)
            
            # Defensive loop protection
            max_steps = 25
            step_count = 0
            
            while current_node and current_node != END and step_count < max_steps:
                step_count += 1
                action = self.graph.nodes[current_node]
                updated = action(current_state)
                if isinstance(updated, dict):
                    current_state.update(updated)

                if current_node in self.graph.conditional_edges:
                    cond_fn, routing_map = self.graph.conditional_edges[current_node]
                    decision = cond_fn(current_state)
                    current_node = routing_map.get(decision, END)
                elif current_node in self.graph.edges:
                    current_node = self.graph.edges[current_node]
                else:
                    break

            return current_state


class AgentState(TypedDict):
    patient_id: str
    conditions: List[str]
    retrieved_context: Optional[str]
    proposed_medication: Optional[str]
    reasoning: Optional[str]
    retries: int
    max_retries: int
    violations: List[str]
    status: str
    final_output: dict


# Lazy global instances for backward compatibility
_GLOBAL_LLM = None
_GLOBAL_RETRIEVER = None

def get_llm_agent() -> LLMClinicalAgent:
    global _GLOBAL_LLM
    if _GLOBAL_LLM is None:
        _GLOBAL_LLM = LLMClinicalAgent()
    return _GLOBAL_LLM

def get_retriever() -> HybridClinicalRetriever:
    global _GLOBAL_RETRIEVER
    if _GLOBAL_RETRIEVER is None:
        _GLOBAL_RETRIEVER = HybridClinicalRetriever()
    return _GLOBAL_RETRIEVER


def retrieve_node(state: AgentState, retriever: HybridClinicalRetriever):
    """Retrieves relevant guideline chunks using FAISS + BM25 RRF (bypassing if already retrieved)."""
    if state.get("retrieved_context"):
        return state

    query = f"Management of patient with {', '.join(state['conditions'])}"
    print(f"\n[Hybrid RAG] Retrieving guidelines for: {query}...")
    context = retriever.retrieve_context(query, top_k=2)
    state["retrieved_context"] = context
    return state


def generate_hypothesis_node(state: AgentState, llm_agent: LLMClinicalAgent):
    attempt = state["retries"] + 1
    print(f"\n[LLaMA Agent] Generating hypothesis (Attempt {attempt})...")
    
    response = llm_agent.generate_recommendation(
        patient_id=state["patient_id"],
        conditions=state["conditions"],
        guidelines=state.get("retrieved_context") or "None",
        violations=state.get("violations", [])
    )
    
    state["proposed_medication"] = response["medication"]
    state["reasoning"] = response["reasoning"]
    print(f"[LLaMA Agent] Proposed Medication: {state['proposed_medication']}")
    print(f"[LLaMA Agent] Clinical Rationale: {state['reasoning']}")
    
    return state


def verify_node(state: AgentState, verifier):
    medication = state.get("proposed_medication", "")
    print(f"\n[HermiT Gate] Verifying '{medication}' against ontology...")
    
    result = verifier.verify_prescription(
        patient_id=state["patient_id"],
        conditions=state["conditions"],
        medications=[medication]
    )
    
    if result["status"] == "PASS":
        print("[HermiT Gate] RESULT: PASSED (Zero Ontological Violations).")
        state["status"] = "VERIFIED"
    else:
        print(f"[HermiT Gate] RESULT: FAILED -> {result['violated_axiom']}")
        state["status"] = "FAILED"
        if "violations" not in state:
            state["violations"] = []
        state["violations"].append(result["violated_axiom"])
        state["retries"] += 1
        
    return state


def route_next_step(state: AgentState):
    if state["status"] == "VERIFIED":
        return "finalize"
    elif state["retries"] >= state.get("max_retries", 3):
        return "escalate"
    else:
        return "retry"


def finalize_node(state: AgentState):
    state["final_output"] = {
        "status": "APPROVED",
        "patient_id": state["patient_id"],
        "recommended_treatment": state["proposed_medication"],
        "clinical_rationale": state["reasoning"],
        "evidence_context": state.get("retrieved_context"),
        "total_attempts": state["retries"] + 1,
        "audit_trail": state.get("violations", [])
    }
    return state


def escalate_node(state: AgentState):
    state["final_output"] = {
        "status": "ESCALATED_FOR_HUMAN_REVIEW",
        "patient_id": state["patient_id"],
        "last_proposed_treatment": state.get("proposed_medication"),
        "reason": "Maximum retry limit reached without resolving formal clinical violations.",
        "audit_trail": state.get("violations", [])
    }
    return state


def build_safecds_graph(verifier, llm_agent: Optional[LLMClinicalAgent] = None, retriever: Optional[HybridClinicalRetriever] = None):
    """
    Builds and compiles the SafeCDS LangGraph state machine.
    Supports dependency injection for high performance and mockability.
    """
    active_llm = llm_agent or get_llm_agent()
    active_retriever = retriever or get_retriever()
    
    workflow = StateGraph(AgentState)
    
    workflow.add_node("retrieve", lambda s: retrieve_node(s, active_retriever))
    workflow.add_node("generate_hypothesis", lambda s: generate_hypothesis_node(s, active_llm))
    workflow.add_node("verify", lambda s: verify_node(s, verifier))
    workflow.add_node("finalize", finalize_node)
    workflow.add_node("escalate", escalate_node)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate_hypothesis")
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