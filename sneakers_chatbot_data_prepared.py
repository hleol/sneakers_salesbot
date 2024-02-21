import gradio as gr
import os

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.langchain_openai import OpenAIEmbeddings


cwd = os.getcwd()

source_file = f"{cwd}/seed/sneakers_sales_chat_data.txt"
with open(source_file) as f:
    sneakers_sales = f.read()

text_splitter = CharacterTextSplitter(
    separator = r'\d+\.',
    chunk_size = 100,
    chunk_overlap  = 0,
    length_function = len,
    is_separator_regex = True,
)

docs = text_splitter.create_documents([sneakers_sales])
db = FAISS.from_documents(docs, OpenAIEmbeddings())
db.save_local("sneakers_sale")

