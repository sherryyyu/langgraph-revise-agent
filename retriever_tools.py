import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from langchain.tools.retriever import create_retriever_tool
from retriever import WatsonxDiscoveryRetriever, NL2SQLRetriever

load_dotenv()

def create_wxd_retriever_tool(name = "wxd_retrieve_documents", description = "Search and return information from documents and images in Watsonx Discovery"):
    es_url = os.getenv("WXD_URL")
    es_user = os.getenv("WXD_USERNAME")
    es_password = os.getenv("WXD_PASSWORD")

    es_client = Elasticsearch(hosts=es_url,
                    ca_certs="http_ca.crt",
                    basic_auth=(es_user, es_password))
    # logger.info(es_client.info())
    retriever = WatsonxDiscoveryRetriever(client=es_client, index_name="multi-modal-rag")
    retriever_tool = create_retriever_tool(
        retriever,
        name,
        description,
    )
    return retriever_tool

def create_sql_retriever_tool(name = "sql_retrieve_documents", description = "Search and return information from SQL database"):
    retriever = NL2SQLRetriever()
    retriever_tool = create_retriever_tool(
        retriever,
        name,
        description,
    )
    return retriever_tool


if __name__ == "__main__":
    create_wxd_retriever_tool()