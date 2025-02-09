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
import json
import requests
import re
import pytesseract
import fitz  # PyMuPDF
from PIL import Image
from pdf2image import convert_from_path
from typing import Dict, List


class MedicalScanPipeline:
    """
    A high-performance, scalable pipeline for processing uploaded medical scan reports.
    It extracts textual data, formats the content, generates context, and predicts key medical insights.
    """

    def __init__(self):
        self.ocr_processor = pytesseract.image_to_string

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
        text = re.sub(r"[^\w\s.,-]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _generate_context(self, text: str) -> str:
        """Generates structured context from formatted medical text."""
        return f"Medical Report Analysis:\n{text}\nEnd of Report."

    def _extract_key_insights(self, context: str) -> Dict:
        """Processes and parses the complete analysis response."""
        raw_response = self._medical_ai(context, task="full_analysis")
        print(f"Raw AI Response: {raw_response}")  # Debug print

        # Check if the response is already in JSON format
        try:
            response_json = json.loads(raw_response)
            if all(k in response_json for k in ["lab_key_points", "predicted_disease", "abnormalities"]):
                return response_json
        except json.JSONDecodeError:
            pass  # Continue with text-based parsing if JSON parsing fails

        # Fallback: Parse the raw text response
        try:
            response = {
                "lab_key_points": self._parse_section(raw_response, "LAB ANALYSIS"),
                "predicted_disease": self._parse_section(raw_response, "DISEASE PREDICTION"),
                "abnormalities": self._parse_section(raw_response, "ABNORMALITY DETECTION")
            }
            return response

        except Exception as e:
            print(f"Parsing Failed: {str(e)}")
            return self._get_fallback_response()

    def _parse_section(self, text: str, section: str) -> List[str]:
        """Extracts and formats specific sections from the AI response."""
        pattern = rf"{section}:\n([\s\S]+?)(?=\n[A-Z ]+\n|$)"  # Extract everything until the next section or end of text
        match = re.search(pattern, text)

        if match:
            extracted_text = match.group(1).strip()

            # Extract both numbered and dash bullet points
            items = re.findall(r"\d+\.\s+(.+)", extracted_text)  # Match lines like "1. Item"
            if not items:
                items = re.findall(r"-\s+(.+)", extracted_text)  # Match lines like "- Item"

            return items

        return []

    def _get_fallback_response(self) -> Dict:
        """Returns a structured fallback response when JSON parsing fails."""
        return {
            "lab_key_points": ["AI processing unavailable"],
            "predicted_disease": ["AI processing unavailable"],
            "abnormalities": ["AI processing unavailable"],
        }

    def _medical_ai(self, text: str, task: str) -> str:
        """Handles AI-based medical analysis using a secure external API execution."""
        from dotenv import load_dotenv
        load_dotenv()

        hitspoint = os.getenv("MEDICAL_AIPOINT")

        def safe_json_load(json_str: str) -> dict:
            try:
                cleaned = json_str.replace("\\\n", "").replace("\\n", "\n")
                return json.loads(cleaned)
            except (json.JSONDecodeError, AttributeError) as e:
                print(f"JSON Error: {str(e)}")
                print(f"Problematic JSON: {json_str}")
                return {}

        hd = safe_json_load(os.getenv("MEDICAL_AI_HEAD").replace("'", '"'))
        data = safe_json_load(os.getenv("AI_BODY").replace("'", '"'))

        if "messages" in data and len(data["messages"]) > 1:
            data["messages"][1]["content"] = f"""
            Task: {task}
            Text: {text}
            REQUIRED RESPONSE FORMAT:
            {{
            "lab_key_points": ["point1", "point2"],
            "predicted_disease": ["diagnosis1", "diagnosis2"], 
            "abnormalities": ["abnormality1", "abnormality2"]
            }}
            """

        response = requests.post(hitspoint, json=data, headers=hd)
        return (
            response.json()
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "AI processing unavailable")
        )


# Usage Example:
if __name__ == "__main__":
    pipeline = MedicalScanPipeline()
    report = pipeline.process_scan("docs/sample_medical_report_2.pdf")
    print(json.dumps(report, indent=4))

