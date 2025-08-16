#!/usr/bin/env python3
"""
Clean version of Atomic Agents framework for testing.
"""

import os
import json
import google.generativeai as genai
from typing import Dict, Any, Optional

def setup_gemini():
    """Setup Gemini API"""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing API key")
    
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config=genai.types.GenerationConfig(
            temperature=0.1,
            max_output_tokens=4096,
            top_p=1,
            top_k=1
        )
    )
    return model

# Global model instance
model = setup_gemini()

def analyze_with_gemini(prompt: str, context: str) -> str:
    """Use Gemini model for analysis"""
    full_prompt = f"{prompt}\n\nContext:\n{context}"
    try:
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def medical_expert_analysis(context: str) -> str:
    """Medical expert agent analysis function"""
    prompt = """You are a Medical Expert Agent specializing in clinical trial analysis.
You have deep knowledge of medical terminology, clinical concepts, and trial procedures.
Your role is to analyze statements from a medical perspective and identify relevant clinical insights.

Steps:
1. Analyze the medical terminology and concepts in the statement
2. Identify key clinical elements and their significance
3. Review the clinical trial data for medical accuracy
4. Assess whether the medical claims align with the trial evidence
5. Provide medical reasoning and clinical context

Provide a clear medical analysis focusing on:
- Medical terminology accuracy
- Clinical relevance and significance
- Alignment with medical evidence in the trial
- Any medical concerns or considerations

End with: MEDICAL_ASSESSMENT: [SUPPORTS/CONTRADICTS/UNCLEAR] based on medical evidence"""
    
    return analyze_with_gemini(prompt, context)

def test_atomic_agents():
    """Test atomic agents functionality"""
    test_context = """
STATEMENT TO ANALYZE: "This trial has 100 participants"

PRIMARY TRIAL DATA:
Title: Test Trial
Phase: Phase 2
Status: Completed
Conditions: Diabetes
Interventions: Drug A vs Placebo
Enrollment: 100 patients
    """
    
    result = medical_expert_analysis(test_context)
    return result

if __name__ == "__main__":
    print("Testing Atomic Agents...")
    result = test_atomic_agents()
    print(f"Result: {result[:200]}...")
    print("✅ Atomic Agents test completed")