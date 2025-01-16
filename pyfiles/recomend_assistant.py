import openai
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from typing import Optional
from PIL import Image
from dotenv import load_dotenv
import io, os
import base64

# Initialize the app
app = FastAPI()

load_dotenv()

api_key = os.getenv("OPEN_AI_API_KEY_KELSEY")
print(api_key)

openai.api_key = api_key

class TextInputData(BaseModel):
    text: str

class ImageInputData(BaseModel):
    image: str  # Base64-encoded image

@app.post("/text-agent")
async def text_agent(input_data: TextInputData):
    text_input = input_data.text

    # Send the text input to the OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for text-based recommendations."},
            {"role": "user", "content": f"Provide recommendations based on the following input:\n{text_input}"}
        ],
        max_tokens=150
    )

    return {"recommendation": response['choices'][0]['message']['content'].strip()}

@app.post("/image-agent")
async def image_agent(input_data: ImageInputData):
    image_data = input_data.image

    # Decode and process the image
    try:
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        # Placeholder for image analysis; replace with your implementation
        image_description = "Image content analysis placeholder."
    except Exception as e:
        return {"error": f"Invalid image data: {str(e)}"}

    # Send the image description to the OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for image-based recommendations."},
            {"role": "user", "content": f"Provide recommendations based on the following image description:\n{image_description}"}
        ],
        max_tokens=150
    )

    return {"recommendation": response['choices'][0]['message']['content'].strip()}

@app.post("/manager")
async def manager(text: Optional[str] = Form(None), file: Optional[UploadFile] = None):
    if text and file:
        return {"error": "Please provide either text or an image, not both."}
    
    if text:
        # Use the text agent
        return await text_agent(TextInputData(text=text))

    if file:
        # Read and encode the image
        try:
            contents = await file.read()
            image_data = base64.b64encode(contents).decode('utf-8')
            return await image_agent(ImageInputData(image=image_data))
        except Exception as e:
            return {"error": f"Failed to process image: {str(e)}"}

    return {"error": "No input provided. Please send either text or an image."}

# Example usage with a client
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
