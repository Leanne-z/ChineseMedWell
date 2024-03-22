"""
This script refers to the dialogue example of streamlit, the interactive generation code of chatglm2 and transformers.
We mainly modified part of the code logic to adapt to the generation of our model.
Please refer to these links below for more information:
    1. streamlit chat example: https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
    2. chatglm2: https://github.com/THUDM/ChatGLM2-6B
    3. transformers: https://github.com/huggingface/transformers
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from dataclasses import asdict

import streamlit as st
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from tools.transformers.interface import GenerationConfig, generate_interactive

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.chains import LLMChain
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

logger = logging.get_logger(__name__)



class InternLM_LLM(LLM):
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    def __init__(self,model,tokenizer):
        # model_path: InternLM 模型路径
        # 从本地初始化模型
        super().__init__()
        self.tokenizer=tokenizer
        self.model=model
        self.model = self.model.eval()

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        # 重写调用函数
        system_prompt = """
            你是中医养生小帮手，熟知中医养生知识，可以提供关于中医养生、中医食疗和自我调养的专业建议和信息，无论是想了解如何通过中医理念改善健康，还是寻找针对特定健康问题的食疗建议，都能提供帮助。
        """

        messages = [(system_prompt, '')]
        response, history = self.model.chat(self.tokenizer, prompt, history=messages)
        return response

    @property
    def _llm_type(self) -> str:
        return "InternLM"

def load_chain(model,tokenizer):
    # 加载问答链
    # 定义 Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="/root/tutorial/lesson_2/model/sentence-transformer")

    # with open("/root/Chinese-Med/data/rag/combine.txt") as f:
        # docs = f.read()

    # text_splitter = RecursiveCharacterTextSplitter(
    # chunk_size=600, chunk_overlap=120)

    # texts = text_splitter.create_documents([docs])

    # bm25_retriever = BM25Retriever.from_documents(texts)
    # bm25_retriever.k =  2

    # 向量数据库持久化路径
    persist_directory = '/root/Chinese-Med/data/rag/vector_db/chroma'

    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
        embedding_function=embeddings
    )

    retriever_chroma=vectordb.as_retriever(search_kwargs={"k": 3})

    # ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, retriever_chroma], weights=[0.4, 0.6])
    


    # 加载自定义 LLM
    llm = InternLM_LLM(model,tokenizer)


    # 定义一个 Prompt Template
    template = """你可以参考以下上下文进行思考，并回答最后的问题。不要表明思考过程，直接返回答案。如果你不知道答案，就说你不知道，不要试图编造答
    案。请提供详细并且结构清晰的回答，并尽量避免简单带过问题。
    {context}
    问题: {question}
    有用的回答:"""

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

    # 运行 chain
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever_chroma, return_source_documents=True,
                                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

    return qa_chain


def on_btn_click():
    del st.session_state.messages


@st.cache_resource
def load_model():
    #/root/traffic_assistant_rag/model/LindseyChang/TRLLM-Model-v2
    #/root/share/model_repos/internlm2-chat-7b
    model = (
        AutoModelForCausalLM.from_pretrained("/root/model/Shanghai_AI_Laboratory/internlm-chat-7b", trust_remote_code=True)
        .to(torch.bfloat16)
        .cuda()
    )
    tokenizer = AutoTokenizer.from_pretrained("/root/model/Shanghai_AI_Laboratory/internlm-chat-7b", trust_remote_code=True)
    return model, tokenizer


def prepare_generation_config():
    with st.sidebar:
        max_length = st.slider("Max Length", min_value=32, max_value=2048, value=2048)
        top_p = st.slider("Top P", 0.0, 1.0, 0.8, step=0.01)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, step=0.01)
        st.button("Clear Chat History", on_click=on_btn_click)
        enable_rag=st.checkbox('RAG检索')
        

    generation_config = GenerationConfig(max_length=max_length, top_p=top_p, temperature=temperature,repetition_penalty=1.002)

    return generation_config,enable_rag


user_prompt = '<|im_start|>user\n{user}<|im_end|>\n'
robot_prompt = '<|im_start|>assistant\n{robot}<|im_end|>\n'
cur_query_prompt = '<|im_start|>user\n{user}<|im_end|>\n\
    <|im_start|>assistant\n'


def combine_history(prompt):
    messages = st.session_state.messages
    meta_instruction = ('You are InternLM (书生·浦语), a helpful, honest, '
                        'and harmless AI assistant developed by Shanghai '
                        'AI Laboratory (上海人工智能实验室).')
    total_prompt = f"<s><|im_start|>system\n{meta_instruction}<|im_end|>\n"
    for message in messages:
        cur_content = message['content']
        if message['role'] == 'user':
            cur_prompt = user_prompt.format(user=cur_content)
        elif message['role'] == 'robot':
            cur_prompt = robot_prompt.format(robot=cur_content)
        else:
            raise RuntimeError
        total_prompt += cur_prompt
    total_prompt = total_prompt + cur_query_prompt.format(user=prompt)
    return total_prompt


def main():
    # torch.cuda.empty_cache()
    print("load model begin.")
    model, tokenizer = load_model()
    qa_chain=load_chain(model, tokenizer)
    print("load model end.")

    user_avator = "doc/imgs/user.png"
    robot_avator = "doc/imgs/robot.png"

    st.title("中医养生助手大语言模型")

    generation_config,enable_rag = prepare_generation_config()

    # enable_rag=st.checkbox('RAG检索')

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message.get("avatar")):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        with st.chat_message("user", avatar=user_avator):
            st.markdown(prompt)
        real_prompt = combine_history(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt, "avatar": user_avator})

        if enable_rag:
            with st.chat_message("robot", avatar=robot_avator):
                message_placeholder = st.empty()
                cur_response=qa_chain({"query": prompt})["result"]
                message_placeholder.markdown(cur_response)
                print(qa_chain({"query": prompt})["source_documents"])

        else:
            with st.chat_message("robot", avatar=robot_avator):
                message_placeholder = st.empty()
                for cur_response in generate_interactive(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=real_prompt,
                    additional_eos_token_id=92542,
                    **asdict(generation_config),
                ):
                    # Display robot response in chat message container
                    message_placeholder.markdown(cur_response + "▌")
                message_placeholder.markdown(cur_response)

        # Add robot response to chat history
        st.session_state.messages.append({"role": "robot", "content": cur_response, "avatar": robot_avator})
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()