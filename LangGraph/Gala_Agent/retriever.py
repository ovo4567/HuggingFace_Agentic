import datasets
from langchain_core.documents import Document

guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split = "train")

docs = [
    Document(
        page_content = "\n".join([
            f"Name: {guest['name']}",
            f"Relation: {guest['relation']}",
            f"Description: {guest["description"]}",
            f"Email: {guest['email']}"
        ]),
        metadata = {"name":guest["name"]}

    )
    for guest in guest_dataset
]

from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import Tool

bm25_retriever = BM25Retriever.from_documents(docs)

# B. Semantic Retriever (FAISS + Embeddings) - Good for: "The scientist", "Electricity expert"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)
semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

def hybrid_extract_text(query: str) -> str:
    """Retrieves detailed guest info using both keyword and semantic search."""
    
    # Get results from BM25 (Top 5)
    bm25_results = bm25_retriever.invoke(query)[:5]
    
    # Get results from Semantic Search (Top 5)
    semantic_results = semantic_retriever.invoke(query)[:5]
    
    # Combine and Rank using Reciprocal Rank Fusion (RRF)
    # RRF Formula: Score = 1 / (k + rank)
    # k is a constant to prevent high ranks from dominating too much (usually 60)
    k = 60
    combined_scores = {}

    def add_to_scores(results, offset):
        for rank, doc in enumerate(results):
            # Use the document's content or metadata as a unique key for deduplication
            doc_id = doc.page_content 
            score = 1 / (k + rank)
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + score

    add_to_scores(bm25_results, 0)
    add_to_scores(semantic_results, 0)

    # Sort documents by their new hybrid score
    sorted_docs = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
    
    # Get the content of the top results
    if sorted_docs:
        # We only want to return the text, not the scores
        top_content = [doc_id for doc_id, score in sorted_docs[:3]]
        return "\n\n".join(top_content)
    else:
        return "No matching guest information found."

# --- 4. Create the Tool ---

guest_info_tool = Tool(
    name="guest_info_retriever",
    func=hybrid_extract_text,
    description="Retrieves detailed information about gala guests using keyword and semantic search (best for names or descriptions)."
)

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

# Generate the chat interface, including the tools
llm = model = ChatOpenAI(
    openai_api_base="http://127.0.0.1:1234/v1",
    openai_api_key="lm-studio",
    model = "local-model"
)

tools = [guest_info_tool]
chat_with_tools = llm.bind_tools(tools)

# Generate the AgentState and Agent graph
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState):
    return {
        "messages": [chat_with_tools.invoke(state["messages"])],
    }

memory = MemorySaver()

## The graph
builder = StateGraph(AgentState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message requires a tool, route to tools
    # Otherwise, provide a direct response
    tools_condition,
)
builder.add_edge("tools", "assistant")
alfred = builder.compile(checkpointer = memory) # add memory to the chatbot so that it remember the chat history

config = {"configurable": {"thread_id": "user_123"}} # This is the "User's ID"

messages = [HumanMessage(content="Tell me about our guest named 'Lady Ada Lovelace'.")]
response = alfred.invoke({"messages": messages}, config=config) # config adds the secion id so that it can remember the past chat

print("🎩 Alfred's Response:")
print(response['messages'][-1].content)

# TURN 1: The first question
print("--- Turn 1 ---")
messages = [HumanMessage(content="Hi Alfred, my name is John.")]
response = alfred.invoke({"messages": messages}, config=config)
print(response['messages'][-1].content)

# TURN 2: A follow-up question (Notice we only send the NEW message!)
print("\n--- Turn 2 ---")
# We do NOT pass the whole history again. LangGraph retrieves it using thread_id!
new_message = [HumanMessage(content="Do you remember my name?")]
response = alfred.invoke({"messages": new_message}, config=config)
print(response['messages'][-1].content)