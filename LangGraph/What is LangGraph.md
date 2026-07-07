### `LangGraph` is a framework developed by LangChain to manage the control flow of applications that integrate an LLM.

## When should we use LangGraph?

When designing AI applications, we face the fundamental trade-off between control and freedom:

- **Freedom** gives your LLM more room to be creative and tackle unexpected problems.
- **Control** allows you to ensure predictable behavior and maintain guardrails.

Code agents in smolagents are very free, it can call multiple tools in a single action step, create their own tools. However, this behavior can make them less predictable and less controllable than a regular Agent working with JSON. In another words, the output might vary for the same query everytime.

`Langchain` shine when you need ""**Control**" over freedom.

```mermaid
graph TD

    %% Node Declarations
    START([● START]):::startEnd
    RouterNode[🔮 Router Node<br/><i>Classifies Inquiry</i>]:::nodeStyle
    RAGNode[📚 Knowledge Retrieval<br/><i>Vector DB Search</i>]:::nodeStyle
    BillingTool[💳 Billing Tool<br/><i>Account Access</i>]:::toolStyle
    DraftNode[✍️ Draft Response<br/><i>LLM Generation</i>]:::nodeStyle
    CriticNode{🧐 Quality Critic<br/><i>Validation Check</i>}:::conditionalStyle
    END([● END]):::startEnd

    %% Flow and Edge Definitions
    START --> RouterNode

    %% Routing Decisions (Conditional Edges)
    RouterNode -->|Technical Support| RAGNode
    RouterNode -->|Billing/Account| BillingTool
    RouterNode -->|General Greeting| DraftNode

    %% Document / Tool Pipelines
    RAGNode --> DraftNode
    BillingTool --> DraftNode

    %% Evaluation Loop
    DraftNode --> CriticNode
    
    %% Critic Routing
    CriticNode -->|Pass: Valid Answer| END
    CriticNode -->|Fail: Missing Context| RAGNode
    CriticNode -->|Fail: Hallucination| DraftNode

    %% Subgraph layout for structural clarity
    subgraph Agentic_Core [LangGraph State Engine]
        RouterNode
        RAGNode
        BillingTool
        DraftNode
        CriticNode
    end
```

Key scenarios where LangGraph excels include:
- Multi-step reasoning processes that need explicit control on the flow
- Applications requiring persistence of state between steps
- Systems that combine deterministic logic with AI capabilities
- Workflows that need human-in-the-loop interventions
- Complex agent architectures with multiple components working together

When you need to design a flow of actions based on the output of each action, and decide what to execute next accordingly, LangGraph is the correct framework.

LangGraph uses a directed graph structure to define the flow of your application:

- **Nodes** represent individual processing steps (like calling an LLM, using a tool, or making a decision)

- **Edges** define the possible transitions between steps

- **State** is user defined and maintained and passed between nodes during execution. When deciding which node to target next, this is the current state that we look at
