import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Pydantic schema for structured output
class ClinicalSuggestion(BaseModel):
    diagnosis_reasoning: str = Field(description="Clinical explanation of the patient's state")
    proposed_medication: str = Field(description="Exact single medication name to prescribe (e.g., Metformin, Insulin, Lisinopril)")

class LLMClinicalAgent:
    def __init__(self, model_name: str = "llama3.2:3b"):
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.1
        )
        self.structured_llm = self.llm.with_structured_output(ClinicalSuggestion)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are a clinical decision support assistant specialized in cardiometabolic multi-morbidity.\n"
             "Analyze the patient's diagnosed conditions and select the most appropriate single medication.\n"
             "Strictly adhere to clinical safety guidelines and avoid any contraindicated medications mentioned in the feedback."),
            ("human", 
             "Patient ID: {patient_id}\n"
             "Known Conditions: {conditions}\n"
             "Previous Safety Violations/Constraints: {violations}\n\n"
             "Provide your diagnosis reasoning and proposed medication.")
        ])

    def generate_recommendation(self, patient_id: str, conditions: list[str], violations: list[str]) -> dict:
        chain = self.prompt | self.structured_llm
        
        violation_text = "\n".join(violations) if violations else "None. This is the first attempt."
        conditions_text = ", ".join(conditions)

        response = chain.invoke({
            "patient_id": patient_id,
            "conditions": conditions_text,
            "violations": violation_text
        })

        return {
            "reasoning": response.diagnosis_reasoning,
            "medication": response.proposed_medication.strip()
        }