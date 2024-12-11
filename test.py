import streamlit as st
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


st.title("DEMO")
st.write("API demo")

select_model = st.text_input('Copy & paste model name...', placeholder=model_name, value=model_name)
temp = st.text_input('Temperature', placeholder="temperature", value=0.4)

llm = ChatOpenAI(
    openai_api_key=openai_key,
    model_name=select_model,
    openai_api_base=base_url,
    temperature=temp,
    streaming=True,
    max_tokens=4000,
    callbacks=[SimpleCallback(que)],
    # model_kwargs={"response_format": {"type": "json_object"}},
)


SYSPROMPT = st.text_area("Write your prompt here...", height=300)

prompt = PromptTemplate.from_template(SYSPROMPT)

chain = LLMChain(llm=llm, prompt=prompt)

st.button("Submit", type="primary")

st.write("LLM RESPONSE: ")
res = st.empty()


if SYSPROMPT or SYSPROMPT != "":
    # res.write(chain.run({'input': ''}))
    txt = ""
    for chunk in chain.run({'input': ''}):
        txt = txt + chunk
        res.write(txt)
