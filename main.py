import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

app = FastAPI()

client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# ========== 启动时加载向量库 ==========
print("正在加载文档并构建向量库...")
loader = TextLoader("docs/产品说明.md", encoding="utf-8")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
print("向量库加载完成！")


class SummarizeRequest(BaseModel):
    text: str

class AskRequest(BaseModel):
    question: str

@app.get("/")
def read_index():
    return FileResponse("static/index.html")

@app.post("/summarize")
def summarize(req: SummarizeRequest):
    try:
        response = client.chat.completions.create(
            model="glm-4.7-flash",
            messages=[
                {"role": "system", "content": "你是一个总结助手，请用一句话总结用户发来的内容。"},
                {"role": "user", "content": req.text}
            ]
        )
        return {"summary": response.choices[0].message.content}
    except Exception as e:
        print("调用大模型出错：", e)
        raise HTTPException(status_code=500, detail="大模型调用失败，请稍后重试。")

@app.post("/ask")
def ask(req: AskRequest):
    try:
        # 1. 检索相关文档
        retrieved = vectorstore.similarity_search(req.question, k=3)
        context = "\n\n".join([doc.page_content for doc in retrieved])

        # 2. 拼 Prompt 调用模型
        response = client.chat.completions.create(
            model="glm-4.7-flash",
            messages=[
                {"role": "system", "content": "基于以下资料回答问题，如果资料里没有答案，就说不知道。\n\n资料：\n" + context},
                {"role": "user", "content": req.question}
            ]
        )
        return {"answer": response.choices[0].message.content}
    except Exception as e:
        print("RAG 出错：", e)
        raise HTTPException(status_code=500, detail="RAG 调用失败，请稍后重试。")