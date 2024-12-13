import time

import streamlit as st
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from PyPDF2 import PdfReader
import docx
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains.llm import LLMChain
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

job_done = "STOP"

openai_key = os.environ.get('OPENAI')
model_name = os.environ.get('GPT_MODEL_4')
base_url = os.environ.get('BASE_URL')

que = asyncio.Queue()


class SimpleCallback(BaseCallbackHandler):
    def __init__(self, q: asyncio.Queue):
        self.q = q

    async def on_llm_start(self, serialized, prompts, **kwargs):
        pass

    async def on_llm_new_token(self, token: str, **kwargs):
        await self.q.put(token)

    async def on_llm_end(self, *args, **kwargs):
        await self.q.put(job_done)

option = st.selectbox(
    "Select your model",
    ("meta-llama/Meta-Llama-3.1-8B-Instruct", model_name, model_name + "-Turbo"),
)

temp = st.text_input('Temperature', placeholder="temperature", value=0.4)

chunk_size = st.number_input('Chunk Size', value=1300)
overlap = st.number_input('Chunk Overlap', value=20)

llm = ChatOpenAI(
    openai_api_key=openai_key,
    model_name=option,
    openai_api_base=base_url,
    temperature=temp,
    streaming=True,
    max_tokens=4000,
    callbacks=[SimpleCallback(que)],
    # model_kwargs={"response_format": {"type": "json_object"}},
)

PROMPT = """
[INST]
As a summary generator, Use the same language used in the text provided below. Respond in tha same language as the text below paragraph. Please provide a clear, engaging summary of the above paragraph. Do not add any other explanation, just write summary without saying greetings or anything. Do no mention anything about yourself or that you have full filled the requirements. Do not mention sentence like 'Here's a summary of ... in bullet points:'
[/INST]

[TextToSummarize] {paragraph} [/TextToSummarize]
"""

SYSPROMPT = st.text_area("Write your prompt here...", height=300, value=PROMPT)

prompt = PromptTemplate.from_template(SYSPROMPT)

chain = LLMChain(llm=llm, prompt=prompt)

final_summary = ""


async def summarize_chunk(chunk, index):
    global final_summary
    tmp = await chain.arun(paragraph=chunk)
    st.write(f"PROCESS: {index}")
    st.write(tmp)
    return tmp


async def process(chunk_list):
    tasks = []

    for i, chunk in enumerate(chunk_list):
        tasks.append(summarize_chunk(chunk, i))

    tmp = await asyncio.gather(*tasks)
    return tmp


st.title("SUMMARY LOGIC")

uploaded_file = st.file_uploader("Upload a file", type=["pdf", "txt", "docx"])

rcts = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

# Check if a file is uploaded
if uploaded_file is not None:
    # Display file details
    st.write("File Details:")
    st.write(f"Filename: {uploaded_file.name}")
    st.write(f"File Type: {uploaded_file.type}")
    st.write(f"File Size: {uploaded_file.size} bytes")

    corpus = ""

    if uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        document = docx.Document(uploaded_file)

        for x, page in enumerate(document.paragraphs):
            # print(f"PAGE {x + 1}")
            corpus += str(page.text).lower() + " "

    elif uploaded_file.type == "application/pdf":
        try:
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    corpus += page_text
                else:
                    st.warning("Some pages may not contain extractable text.")
        except Exception as e:
            st.error(f"Error reading PDF: {e}")

    # Example: If it's a text file, display content
    elif uploaded_file.type == "text/plain":
        st.write("File Content:")
        corpus = uploaded_file.read().decode("utf-8")

    st.text_area("File Content", corpus, height=300)

    split_text = rcts.split_text(corpus)

    st.write("Chunked Document:")
    st.write(split_text)

    final_summary = " ".join(asyncio.run(process(split_text)))

    st.write("FINAL SUMMARY: ")

    st.write(chain.run(paragraph=final_summary))

