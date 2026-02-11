from dotenv import load_dotenv; load_dotenv(".env")

import os
import rich
import asyncio
import pickle
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from langchain_neo4j import Neo4jGraph
from langchain_neo4j.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer

from models.chat_models import ChatDeepSeek, ChatDeepSeekModelName
from models.embeddings import AzureOpenAIEmbeddings, AzureOpenAIEmbeddingModelName


async def main1():
    graph = Neo4jGraph(
        url="neo4j+s://b999a1b6.databases.neo4j.io",
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        database="neo4j",
    )

    graph_transformer = LLMGraphTransformer(
        llm=ChatDeepSeek(ChatDeepSeekModelName.deepseek_chat),
    )

    if os.path.exists("db/graph_documents.pkl"):
        with open("db/graph_documents.pkl", "rb") as f:
            graph_documents = pickle.load(f)
    else:
        with open("assets/documents/company.txt", "r") as f:
            content = f.read()
        documents =[Document(page_content=content, metadata={"source": "company.txt"})]
        graph_documents = graph_transformer.convert_to_graph_documents(documents)

        with open("db/graph_documents.pkl", "wb") as f:
            pickle.dump(graph_documents, f)

        graph.add_graph_documents(graph_documents)

    cypher_chain = GraphCypherQAChain.from_llm(
        graph=graph,
        cypher_llm=ChatDeepSeek(ChatDeepSeekModelName.deepseek_chat),
        qa_llm=ChatDeepSeek(ChatDeepSeekModelName.deepseek_chat),
        validate_cypher=True,
        verbose=True,
        allow_dangerous_requests=True
    )

    response = cypher_chain.invoke({
        "query": "西红柿炒鸡蛋怎么弄"
    })
    print(response)

async def main():

    if os.path.exists("db/splits_documents.pkl"):
        with open("db/splits_documents.pkl", "rb") as f:
            splits_documents = pickle.load(f)
    else:
        with open("assets/documents/company.txt", "r") as f:
            content = f.read()
        documents =[Document(page_content=content, metadata={"source": "company.txt"})]

        chunk_size = 250
        chunk_overlap = 30
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        splits_documents = text_splitter.split_documents(documents)

        with open("db/splits_documents.pkl", "wb") as f:
            pickle.dump(splits_documents, f)

    embeddings = AzureOpenAIEmbeddings(AzureOpenAIEmbeddingModelName.text_embedding_3_small)
    vectorstore = Milvus.from_documents(
        documents=splits_documents,
        collection_name="Test",
        embedding=embeddings,
        connection_args={
            "uri": "https://in03-c6f34b5163152ab.serverless.aws-eu-central-1.cloud.zilliz.com",
            "user": os.getenv("MILVUS_USERNAME"),
            "password": os.getenv("MILVUS_PASSWORD"),
        },
        drop_old=True,
    )

    prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise:
        Question: {question} 
        Context: {context} 
        Answer: 
        """,
        input_variables=["question", "context"],
    )

    # 数据预处理
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    llm = ChatDeepSeek(ChatDeepSeekModelName.deepseek_chat)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

    # 运行时：用 question 检索 → 得到 context，再和 question 一起填入 prompt
    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    response = rag_chain.invoke({
        "question": "我的知识库中都有哪些公司信息",
    })
    print(response)

    

if __name__ == "__main__":
    asyncio.run(main())
