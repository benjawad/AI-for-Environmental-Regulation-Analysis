from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import re
import torch
import json
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import PyPDF2
from docx import Document

# Initialisation du modèle
model_name = "deepset/deberta-v3-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
device = 0 if torch.cuda.is_available() else -1

extractor = pipeline(
    "question-answering", 
    model=model, 
    tokenizer=tokenizer,
    device=device
)

# Define confidence threshold
THRESHOLD = 0.1

# Updated inference rules
inference_rules = {
    "sector": {
        "patterns": ["sector", "domain", "output"],
        "infer": lambda text: ("Climate Tech" if "geodata" in text.lower() 
                             else "Data Infrastructure" if "ZB" in text 
                             else "Information not found")
    },
    "customer": {
        "patterns": ["customer", "client", "sponsor"],
        "extract": r"Sponsor:\s*(.*?)\s*\(",
        "clean": lambda x: x.split("[")[0].strip()
    },
    "technology": {
        "patterns": ["technology"],
        "clean": lambda x: re.sub(r"\([^)]*\)", "", x).strip()
    },
    "duration": {
        "patterns": ["duration", "timeline"],
        "infer": lambda text: "2-3 years (estimated from capacity scaling 5T/week → 75T/day)" if "pilot" in text.lower() else "Duration not specified"
    },
    "safety": {
        "patterns": ["safety", "concern"],
        "infer": lambda text: "High risk: Requires special handling for flammable and toxic materials" 
                              if ("flammable" in text.lower() and "toxic" in text.lower()) 
                              else "Safety concerns not specified"
    },
    "location": {
        "patterns": ["location", "pilot plant"],
        "extract": r"Pilot Plant Location:\s*(.*?)\s*\n",
        "clean": lambda x: x.strip()
    },
    "capacity": {
        "patterns": ["production capacity", "capacity"],
        "extract": r"Production Capacity:\s*(.*?)\s*\n",
        "clean": lambda x: x.strip()
    }
}

def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file):
    """Extract text from DOCX file"""
    doc = Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

@csrf_exempt
def analyze_project(request):
    if request.method == 'POST':
        project_text = ""
        
        # Check if a file was uploaded
        if 'project_file' in request.FILES:
            file = request.FILES['project_file']
            file_extension = file.name.split('.')[-1].lower()
            
            if file_extension == 'pdf':
                project_text = extract_text_from_pdf(file)
            elif file_extension in ['doc', 'docx']:
                project_text = extract_text_from_docx(file)
            else:
                return JsonResponse({'error': 'Unsupported file format'}, status=400)
        else:
            project_text = request.POST.get('project_description', '')
        
        if not project_text:
            return JsonResponse({'error': 'No text provided'}, status=400)
        
        # French questions
        questions = [
            "Quel est le secteur du projet?",
            "Où est située l'installation pilote?",
            "Sur quelle technologie le projet est-il basé?",
            "Quelle est la durée estimée du projet?",
            "Qui est le client?",
            "Quelles sont les préoccupations en matière de sécurité?",
            "Quelle est la capacité de production?"
        ]
        
        results = {}
        for question in questions:
            results[question] = get_enhanced_answer(question, project_text)
        
        return JsonResponse(results)
        
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)

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
    except Exception as e:
        return f"Error processing question: {str(e)}"
    
    answer = result["answer"].strip()
    score = result["score"]
    
    # Apply cleaning if available for this question type
    for key, rule in inference_rules.items():
        if any(pattern in question.lower() for pattern in rule["patterns"]) and "clean" in rule:
            answer = rule["clean"](answer)
    
    # Handle low-confidence results
    if score < THRESHOLD or not answer or answer == "[CLS]":
        return "Information not found in document"
    
    return answer

def home(request):
    return render(request, 'jesa_app/home.html')