import os
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


ht_toekn = os.getenv("HF_TOKEN")
login(token=ht_toekn)

# Load the Mistral 7B model and tokenizer

model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model in 8-bit precision
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)


# Set up the text generation pipeline
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)
















# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.llms import HuggingFaceLLM
# from langchain.chains import RetrievalQA
# from transformers import AutoTokenizer, AutoModel


# huggingface_llm = HuggingFaceLLM(model_name="gpt2")

# # Create a chain (prompting a model to generate text)
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain

# # Load the tokenizer and model
# model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Use an appropriate model
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)

# # Create embeddings using the HuggingFace model
# embeddings = HuggingFaceEmbeddings(model=model, tokenizer=tokenizer)

