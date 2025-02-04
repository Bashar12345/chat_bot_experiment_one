"""  
install the pip packages 
pip install openai  pytesseract fitz pdf2image pillow python-dotenv

then include the MEDICAL_AIPOINT and MEDICAL_AI_HEAD in the .env file

"""





"""
MEDICAL_AIPOINT= https://api.openai.com/v1/chat/completions

MEDICAL_AI_HEAD={"Authorization": "Bearer sk", "Content-Type": "application/json"}
AI_BODY={"model": "gpt-4", "messages": [{"role": "system", "content": "You are a highly advanced medical AI assistant specializing in analyzing medical reports. "
                "Your goal is to extract structured insights, identify lab result abnormalities, predict potential diseases, prioritize tasks that require follow-up, "
                "and detect any anomalies in the given medical context. "
                "Ensure your responses are detailed, evidence-based, and formatted in a structured manner."
                "\n\n"
                "Follow these rules:"
                "\n- Extract key lab report findings and highlight any critical values."
                "\n- Identify and list potential diseases based on the provided data."
                "\n- Detect any abnormal indicators of sickness and provide possible explanations."
                "\n- Respond with concise yet informative medical explanations."
                "\n- Format responses using bullet points for clarity."
            }, {"role": "user", "content": "Task: {task}\\nText: {text}"}]}




"""

AI_BODY='{"model":"gpt-4","messages":[{"role":"system","content":"As a medical AI specialist, analyze texts in 3 phases:\n1. LAB ANALYSIS: Extract and format key lab values with reference ranges\n2. DISEASE PREDICTION: Identify potential diagnoses using clinical guidelines\n3. ABNORMALITY DETECTION: Flag significant deviations with explanations\n\nRespond with JSON structure:\n{\n  \"labs\": [\"item: value (range)\"],\n  \"diagnoses\": [\"condition (confidence%)\"],\n  \"abnormalities\": [\"finding: explanation\"]\n}"},{"role":"user","content":"TASK_TYPE: {task}\nREPORT: {text}"}]}'





"You are a highly advanced medical AI assistant specializing in analyzing medical reports. "
            "Your goal is to prioritize the requested task first while ensuring the response is structured, evidence-based, and clear."
            "\n\n"
            "🔹 **Task Prioritization Order:**"
            "\n1️⃣ **Focus on the specific task requested:** Extract lab findings, predict diseases, or analyze abnormalities."
            "\n2️⃣ **Use extracted lab results and patient symptoms to support the analysis.**"
            "\n3️⃣ **If necessary, provide further context based on the available medical data.**"
            "\n\n"
            "🔹 **Task Definitions:**"
            "\n- `extract_lab_values`: Identify key lab report findings (e.g., abnormal glucose, low hemoglobin)."
            "\n- `predict_disease`: Determine possible diseases based on lab results and symptoms."
            "\n- `detect_abnormalities`: Highlight any irregularities in the patient's medical data."
            "\n\n"
            "🔹 **Response Format:**"
            "\n- Prioritize the requested task above all else."
            "\n- Provide results in a structured format."
            "\n- Use bullet points for clarity."










import os
import re ,requests
import json
import pytesseract
import fitz  # PyMuPDF
# import torch
from PIL import Image
from pdf2image import convert_from_path
# from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, List
import importlib.util

class MedicalScanPipeline:
    """
    A high-performance, scalable pipeline for processing uploaded medical scan reports.
    It extracts textual data, formats the content, generates context, and predicts key medical insights.
    """
    
    def __init__(self, disease_model: str = "bert-base-uncased", abnormality_model: str = "distilbert-base-uncased"):
        self.ocr_processor = pytesseract.image_to_string
        # self.disease_pipeline = self._load_model_pipeline(disease_model)
        # self.abnormality_pipeline = self._load_model_pipeline(abnormality_model)
        self.ai_client = self._initialize_ai_client()
    
    # @staticmethod
    # def _load_model_pipeline(model_name: str):
    #     """Loads a transformer model for classification tasks."""
    #     tokenizer = AutoTokenizer.from_pretrained(model_name)
    #     model = AutoModelForSequenceClassification.from_pretrained(model_name)
    #     return pipeline("text-classification", model=model, tokenizer=tokenizer)
    
    def _initialize_ai_client(self):
        """Dynamically loads the AI client to obscure OpenAI dependency."""
        spec = importlib.util.find_spec("openai")
        if spec is None:
            return None
        ai_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ai_module)
        return ai_module.ChatCompletion
    
    def process_scan(self, file_path: str) -> Dict:
        """Processes a scanned medical document (PDF or image) and returns structured medical insights."""
        extracted_text = self._extract_text(file_path)
        formatted_text = self._format_text(extracted_text)
        context = self._generate_context(formatted_text)
        key_insights = self._extract_key_insights(context)
        return key_insights
    
    def _extract_text(self, file_path: str) -> str:
        """Extracts raw text from an uploaded scanned medical document (PDF or image)."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")
        
        text = ""
        if file_path.lower().endswith(".pdf"):
            text = self._extract_text_from_pdf(file_path)
        else:
            text = self.ocr_processor(Image.open(file_path))
        
        return text.strip()
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extracts text from a PDF using PyMuPDF, with a fallback to OCR if necessary."""
        extracted_text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                extracted_text += page.get_text("text") + "\n"
        
        if not extracted_text.strip():  # If no text is found, use OCR
            images = convert_from_path(file_path)
            extracted_text = "\n".join([self.ocr_processor(img) for img in images])
        
        return extracted_text.strip()
    
    def _format_text(self, text: str) -> str:
        """Cleans and formats extracted text for further processing."""
        text = re.sub(r'[^\w\s.,]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _generate_context(self, text: str) -> str:
        """Generates structured context from formatted medical text."""
        return f"Medical Report Analysis:\n{text}\nEnd of Report."
    
    def _extract_key_insights(self, context: str) -> Dict:
        """Extracts key insights including lab results, predicted diseases, and abnormalities."""
        lab_key_points = self._extract_lab_key_points(context)
        predicted_disease = self._predict_disease(context)
        abnormality_analysis = self._analyze_abnormalities(context)
        
        return {
            "lab_key_points": lab_key_points,
            "predicted_disease": predicted_disease,
            "abnormalities": abnormality_analysis
        }
    
    def _extract_lab_key_points(self, context: str) -> List[str]:
        """Extracts key points from the medical lab report using AI."""
        return self._medical_ai(context, task="extract_lab_values")
    
    def _predict_disease(self, context: str) -> str:
        """Predicts potential diseases based on the provided medical context using AI."""
        return self._medical_ai(context, task="predict_disease")
    
    def _analyze_abnormalities(self, context: str) -> str:
        """Analyzes the medical text for any abnormal indicators of sickness using AI."""
        return self._medical_ai(context, task="detect_abnormalities")
    
    def _medical_ai(self, text: str, task: str) -> str:
        """Handles AI-based medical analysis using a secure external API execution."""
        from dotenv import load_dotenv
        load_dotenv()
        
        hitspoint = os.getenv("MEDICAL_AIPOINT")
        
        # Debug: Print raw environment values
        def safe_json_load(json_str: str) -> dict:
            try:
                # Handle Windows-style line continuations
                cleaned = json_str.replace('\\\n', '').replace('\\n', '\n')
                return json.loads(cleaned)
            except (json.JSONDecodeError, AttributeError) as e:
                print(f"JSON Error: {str(e)}")
                print(f"Problematic JSON: {json_str}")
                return {}

        hd = safe_json_load(os.getenv("MEDICAL_AI_HEAD").replace("'", '"'))
        data = safe_json_load(os.getenv("AI_BODY").replace("'", '"'))

        # Replace placeholders
        if "messages" in data and len(data["messages"]) > 1:
            data["messages"][1]["content"] = f"Task: {task} , Text: {text}"
            
        
        response = requests.post(hitspoint, json=data, headers=hd)
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "AI processing unavailable")

# Usage Example:
if __name__ == "__main__":
    pipeline = MedicalScanPipeline()
    report = pipeline.process_scan("docs/sample_medical_report_2.pdf")
    print(json.dumps(report, indent=4))