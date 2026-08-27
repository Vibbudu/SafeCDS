from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class ClinicalSuggestion(BaseModel):
    diagnosis_reasoning: str = Field(description="Clinical explanation grounding the choice in guidelines and safety rules.")
    proposed_medication: str = Field(description="Exact medication name to prescribe (e.g., Amlodipine, Insulin).")

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
             "Strictly adhere to the provided clinical guidelines and all ontological feedback constraints."),
            ("human", 
             "Patient ID: {patient_id}\n"
             "Known Conditions: {conditions}\n\n"
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
            "reasoning": response.diagnosis_reasoning,
            "medication": response.proposed_medication.strip()
        }