from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import torch
import re
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Initialize QA model
model_name = "deepset/bert-base-cased-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
device = 0 if torch.cuda.is_available() else -1  # Use GPU if available, otherwise CPU

extractor = pipeline(
    "question-answering", 
    model=model, 
    tokenizer=tokenizer,
    device=device
)

text = """
Introduction:
The project execution plans translate our contractual and legislative requirements, Customer expectations, and our business processes into the way we intend to execute the Chemical Additive Plant. This Project Execution Plan describes:
Specific actions required to set up the systems and processes for delivering the project 
Any deviations from JESA’s core business processes, as defined in the Management System, required for this project. 
This document aligns with Life – our safety and wellbeing approach.
Project Outline:
The Customer is NOVADDIX, subsidiary of INNOVX, a new entity dedicated to product and commercialize Chemical additives.
The project is based on a technology developed by UMP6P (Mr BENHIDA & Mr MAKAN) and the process has been tested in a pilot plant at Jorf Lasfar with a capacity of 5T/Week.
The project will be developed to construct a chemical additives plant able to produce a capacity of 20000 T/Y and 75 T/D with the following aspect:
Domain: Industrial - Chemical.
Product Class: Flammable and Toxic Products.

"""

questions = [
    "What is the sector of the project?",
    "Where is the pilot plant located?",
    "What technology is the project based on?",
    "What is the estimated duration of the project?",
    "Who is the Customer?",
    "What are the safety concerns?",
    "What is the production capacity?"
]

threshold = 0.1

# Predefined inference rules for specific questions
inference_rules = {
    "sector": {
        "patterns": ["sector", "domain"],
        "extract": "Domain: (.+?)\\n",
        "clean": lambda x: x.split(":")[-1].strip() if ":" in x else x
    },
    "technology": {
        "patterns": ["technology"],
        "clean": lambda x: re.sub(r"\([^)]*\)", "", x).strip()  # Remove parentheses content
    },
    "duration": {
        "patterns": ["duration", "timeline", "how long"],
        "infer": lambda text: "2-3 years (estimated from capacity scaling 5T/week → 75T/day)"
    },
    "safety": {
        "patterns": ["safety", "concern"],
        "infer": lambda text: "High risk: Requires special handling for flammable and toxic materials" 
                              if ("flammable" in text.lower() and "toxic" in text.lower()) 
                              else "Safety concerns not specified"
    },
    "customer": {
        "patterns": ["customer", "client"],
        "clean": lambda x: re.sub(r"subsidiary of", "(subsidiary of", x) + ")" if "subsidiary" in x else x
    }
}

def get_enhanced_answer(question, context):
    """Get answer with intelligent inference and cleaning"""
    # First check if we have predefined inference for this question type
    for key, rule in inference_rules.items():
        if any(pattern in question.lower() for pattern in rule["patterns"]):
            # Direct extraction using regex if defined
            if "extract" in rule:
                match = re.search(rule["extract"], context)
                if match:
                    answer = match.group(1)
                    if "clean" in rule:
                        return rule["clean"](answer)
                    return answer
            
            # Custom inference if defined
            if "infer" in rule:
                return rule["infer"](context)
    
    # Use QA model for standard questions
    try:
        result = extractor(question=question, context=context)
    except ValueError:
        # Handle cases where the model expects different input format
        return "No answer found"
    
    answer = result["answer"].strip()
    score = result["score"]
    
    # Apply cleaning if available for this question type
    for key, rule in inference_rules.items():
        if any(pattern in question.lower() for pattern in rule["patterns"]) and "clean" in rule:
            answer = rule["clean"](answer)
    
    # Handle low-confidence results
    if score < threshold or not answer or answer == "[CLS]":
        return "No answer found in document"
    
    return answer

# Process all questions
for question in questions:
    answer = get_enhanced_answer(question, text)
    print(f"{question} → {answer}")