from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# 1. 加载文档
loader = TextLoader("docs/产品说明.md", encoding="utf-8")
docs = loader.load()

# 2. 切块
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. 向量化并存入 Chroma
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

# 4. 检索
query = "怎么退货"
retrieved = vectorstore.similarity_search(query, k=3)
context = "\n\n".join([doc.page_content for doc in retrieved])

# 5. 拼 Prompt 调用模型
llm = ChatOpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    model="glm-4.7-flash"
)

prompt = ChatPromptTemplate.from_template(
    "基于以下资料回答问题，如果资料里没有答案，就说不知道。\n\n资料：\n{context}\n\n问题：{question}"
)
chain = prompt | llm
result = chain.invoke({"context": context, "question": query})
print(result.content)