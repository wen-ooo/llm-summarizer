from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

class SummarizeRequest(BaseModel):
    text: str

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
        summary = response.choices[0].message.content
        return {"summary": summary}
    except Exception as e:
        print("调用大模型出错：", e)
        raise HTTPException(status_code=500, detail="大模型调用失败，请稍后重试。")