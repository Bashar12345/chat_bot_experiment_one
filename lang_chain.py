from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
import json


def process_news(json_content):
    # Extract relevant content
    news_context = " ".join(
        f"{article['headline']} - {article['content']} (Source: {article['source']})"
        for article in json_content
    )

    # Use LangChain and Ollama to generate a response
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run({"news_data": news_context})

    return result



# Initialize the Ollama LLM
llm = Ollama(model="llama3")

template = """
You are a news summarizer. Based on the following context, provide a concise summary and list the references.

Context: {news_data}

Output format:
output: [Summary]
reference: [List of sources]
"""
prompt = PromptTemplate(template=template, input_variables=["news_data"])



result = llm.invoke(prompt)
print(result)