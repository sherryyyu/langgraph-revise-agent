import os
import pprint
from pydantic import BaseModel, Field
from loguru import logger

from typing import Annotated, Sequence, Literal
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser


from langchain_ibm import ChatWatsonx

from langgraph.prebuilt import tools_condition
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langchain_core.tools import Tool

from langchain_community.utilities import GoogleSerperAPIWrapper
from retriever_tools import create_wxd_retriever_tool, create_sql_retriever_tool

PROJECT_ID = os.getenv("PROJECT_ID")
wx_url = "https://us-south.ml.cloud.ibm.com"
# llm_id = "meta-llama/llama-3-2-90b-vision-instruct"
llm_id = "meta-llama/llama-3-3-70b-instruct"


# Tools

wxd_retriever_tool = create_wxd_retriever_tool("sustainability_document_search", "useful for searching sustainability reports")
serper_search = GoogleSerperAPIWrapper()
serper_search_tool = Tool(
        name="news_search",
        func=serper_search.run,
        description="useful for when you need to find out news and info of the company via web search",
    )

sql_retriever_tool = create_sql_retriever_tool("sustainability_statistics_search", "useful for searching sustainability statistics for suppliers such as scope 1 and scope 2 carbon emmisions and revenue.")
tools = [wxd_retriever_tool, serper_search_tool, sql_retriever_tool]


# Define the agent's memory
class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]


# Edges
def grade_documents(state) -> Literal["generate", "agent"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    logger.info("---CHECK RELEVANCE---")

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 5000,
    "min_new_tokens": 1,
    }

    model = ChatWatsonx(
        model_id = llm_id,
        url=wx_url,
        project_id=PROJECT_ID,
        params=parameters,
    )

    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        logger.info("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        logger.info("---DECISION: DOCS NOT RELEVANT---")
        return "agent"


# Nodes
def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    logger.info("---CALL AGENT---")
    messages = state["messages"]
    parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 5000,
    "min_new_tokens": 1,
    }
    query = messages[0]

    model = ChatWatsonx(
        model_id = llm_id,
        url=wx_url,
        project_id=PROJECT_ID,
        params=parameters,
    )

    tools_used = []
    for m in messages:
        if hasattr(m, 'tool_call_id'):
            logger.info(f"I have tried tool: {m.name}")
            tools_used.append(m.name)

    unused_tools = [t for t in tools if t.name not in tools_used]    
    model = model.bind_tools(unused_tools)
    logger.debug(f"unused tools: {[u.name for u in unused_tools]}")
    response = model.invoke([query])
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """
    logger.info("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    # Prompt
    prompt = ChatPromptTemplate([
    ("user", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If there is no context, just answer the question to be best of your knowledge. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:")
    ])

    # LLM
    parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 5000,
    "min_new_tokens": 1,
    }

    llm = ChatWatsonx(
        model_id = llm_id,
        url=wx_url,
        project_id=PROJECT_ID,
        params=parameters,
    )

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}


def query_multi_source(user_query:str):
    # Define a new graph
    workflow = StateGraph(AgentState)

    # Define the nodes we will cycle between
    workflow.add_node("agent", agent)  # agent

    retrieve = ToolNode(tools)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_edge(START, "agent")

    workflow.add_conditional_edges(
        "agent",
        # Assess agent decision
        tools_condition,
        {
            # Translate the condition outputs to nodes in our graph
            "tools": "retrieve",
            END: "generate",
        },
    )

    # Edges taken after the `action` node is called.
    workflow.add_conditional_edges(
        "retrieve",
        # Assess agent decision
        grade_documents,
    )
    workflow.add_edge("generate", END)


    # Compile
    graph = workflow.compile()

    inputs = {"messages": [("user", user_query)]}

    for output in graph.stream(inputs):
        for key, value in output.items():
            logger.debug(f"Output from node '{key}':")
            logger.debug("---")
            logger.debug(value)
        logger.debug("\n---\n")

    return value['messages'][0]