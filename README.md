# 大模型总结器

一个基于 FastAPI + 智谱 GLM 的大模型应用，支持：
- 文本总结
- RAG 知识库问答

## 技术栈
FastAPI、LangChain、ChromaDB、智谱 GLM

## 运行方式
1. 安装依赖：`pip install -r requirements.txt`
2. 创建 `.env` 文件，填入 `ZHIPU_API_KEY=你的key`
3. 启动：`fastapi dev main.py`
4. 浏览器访问 `http://127.0.0.1:8000`
