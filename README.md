# 大模型应用（llm-summarizer）

一个基于 FastAPI + 智谱 GLM 的大模型应用，支持：
- **文本总结**：输入一段文字，生成一句话总结
- **知识库问答（RAG）**：基于本地文档回答问题

## 技术栈
- FastAPI：Web 框架
- LangChain + ChromaDB：RAG 检索
- 智谱 GLM：大模型
- Docker：一键部署

## 本地运行
1. 安装依赖：`pip install -r requirements.txt`
2. 创建 `.env` 文件，填入 `ZHIPU_API_KEY=你的key`
3. 启动：`fastapi dev main.py`
4. 浏览器访问 `http://127.0.0.1:8000`

## Docker 运行
docker build -t llm-summarizer .
docker run -p 8000:8000 --env-file .env llm-summarizer

## 项目结构
- `main.py`：FastAPI 服务，包含 `/summarize` 和 `/ask` 两个接口
- `static/index.html`：前端页面
- `docs/`：知识库文档
- `rag_test.py`：RAG 独立测试脚本
- `Dockerfile`：Docker 构建配置
