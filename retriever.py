"""Wrapper around databases."""

from __future__ import annotations
import os
from typing import Any, List


from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.utilities import SQLDatabase
from langchain_ibm import ChatWatsonx
from langchain.chains import create_sql_query_chain


class WatsonxDiscoveryRetriever(BaseRetriever):
    """Watsonx Discovery (Elasticsearch OEM) Retriever

    To connect to an Elasticsearch instance that requires login credentials,
    use the Elasticsearch URL format https://username:password@es_host:9243. 

    You can obtain your Elastic Cloud URL and login credentials by logging in to the
    Elastic Cloud console at https://cloud.elastic.co, selecting your deployment, and
    navigating to the "Deployments" page.

    To obtain your Elastic Cloud password for the default "elastic" user:

    1. Log in to the Elastic Cloud console at https://cloud.elastic.co
    2. Go to "Security" > "Users"
    3. Locate the "elastic" user and click "Edit"
    4. Click "Reset password"
    5. Follow the prompts to reset the password

    The format for Elastic Cloud URLs is
    https://username:password@cluster_id.region_id.gcp.cloud.es.io:9243.
    """

    client: Any
    """Elasticsearch client."""
    index_name: str
    """Name of the index to use in Elasticsearch."""


    def _get_relevant_documents(
        self, query:str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        '''Make a dense vector based knn query to Watsonx Discovery, 
        using the ntfloat__multilingual-e5-large model and retrieve 3 
        documents.'''
        knn_query={
            "field": "embedding",
            "query_vector_builder": {
                "text_embedding": {
                    "model_id": "intfloat__multilingual-e5-large",
                    "model_text": query,
                }
            },
            "k": 3,
            "num_candidates": 10
        }
        res = self.client.search(
            index=self.index_name,
            knn=knn_query,
        )

        docs = []
        for r in res["hits"]["hits"]:
            docs.append(Document(page_content=r["_source"]["content"]))
        return docs
    

class NL2SQLRetriever(BaseRetriever):

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
            ) -> List[Document]:
        """
    Retrieve relevant documents based on a natural language query from a postgres database.

    Args:
        query (str): The query to search for relevant documents.

    Returns:
        List[Document]: A list of relevant documents retrieved based on the query.
    """

        llm = ChatWatsonx(
            model_id="meta-llama/llama-3-3-70b-instruct",
            url="https://us-south.ml.cloud.ibm.com",
            project_id=os.getenv("PROJECT_ID"),
            params={
                "decoding_method": "greedy",
                "max_new_tokens": 1000,
                "min_new_tokens": 1,
            },
        )


        db_uri = os.getenv("PG_URI")
        db = SQLDatabase.from_uri(db_uri)
        chain = create_sql_query_chain(llm, db)
        response = chain.invoke({"question": query})

        return [Document(page_content=response)]
        