from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class ClinicalSuggestion(BaseModel):
    clinical_rationale: str = Field(
        description="A detailed step-by-step clinical justification explaining why this medication was chosen and why alternatives were avoided based on the provided guidelines and safety constraints."
    )
    proposed_medication: str = Field(
        description="Exact medication name to prescribe (e.g., Insulin, Amlodipine)."
    )

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
             "Analyze the patient's conditions and relevant clinical guideline excerpts to select the safest medication.\n"
             "CRITICAL: Base your reasoning ONLY on the patient's listed Known Conditions. Do not assume or hallucinate conditions (like CKD) if they are not explicitly in the patient's record.\n"
             "Strictly adhere to clinical guidelines and all ontological feedback constraints."),
            ("human", 
             "Patient ID: {patient_id}\n"
             "Explicitly Diagnosed Conditions: {conditions}\n\n"
             "Clinical Guidelines & Evidence:\n{guidelines}\n\n"
             "Previous Safety Violations/Constraints:\n{violations}\n\n"
             "Provide your clinical rationale and proposed medication.")
        ])

    def generate_recommendation(self, patient_id: str, conditions: list[str], guidelines: str, violations: list[str]) -> dict:
        chain = self.prompt | self.structured_llm
        
        violation_text = "\n".join(violations) if violations else "None. Initial attempt."
        conditions_text = ", ".join(conditions)

        response = chain.invoke({
            "patient_id": patient_id,
            "conditions": conditions_text,
            "guidelines": guidelines,
            "violations": violation_text
        })

        return {
            "reasoning": response.clinical_rationale,
            "medication": response.proposed_medication.strip()
        }   