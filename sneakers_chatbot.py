import gradio as gr

from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI

def initialize_sales_bot(vector_store_dir: str="sneakers_sale"):
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings())
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    global SALES_BOT
    SALES_BOT = RetrievalQA.from_chain_type(llm,
                                            retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                      search_kwargs={"score_threshold":
                                                                                    0.8}))
    SALES_BOT.return_source_documents = True

    return SALES_BOT

def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    enable_chat = True

    ans = SALES_BOT({"query": message})

    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    else:
        return "这个问题我要问问领导"

def launch_gradio():
    demo = gr.ChatInterface(fn=sales_chat, title="球鞋销售",
                            chatbot=gr.Chatbot(height=600))

    demo.launch(share=True, server_name = "0.0.0.0")

if __name__ == "__main__":
    initialize_sales_bot()
    launch_gradio()
