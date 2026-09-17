<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=30&duration=2500&pause=800&color=00D4AA&center=true&vCenter=true&width=850&lines=Graphiti+%2B+Zep+%E2%80%94+Temporal+Knowledge+Graphs;Local+AI+Agent+Memory+with+Ollama+%2B+Neo4j;Zero+API+Costs+%7C+100%25+Self-Hosted;Bi-Temporal+Reasoning+%7C+Automatic+Fact+Invalidation" alt="Typing SVG" />

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Graphiti](https://img.shields.io/badge/Graphiti-Core_0.30.2-00D4AA?style=for-the-badge&logo=graphql&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-6C63FF?style=for-the-badge&logo=graphql&logoColor=white)
![Neo4j](https://img.shields.io/badge/Neo4j-5.x-008CC1?style=for-the-badge&logo=neo4j&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-phi3:mini-000000?style=for-the-badge&logo=ollama&logoColor=white)
![Cost](https://img.shields.io/badge/API_Cost-$0.00_Free-brightgreen?style=for-the-badge)

<br/>

```text
  ┌────────────────────────────────────────────────────────────────────────┐
  │   🧠 TEMPORAL KNOWLEDGE GRAPH FOR AI AGENTS (CONTEXT | PERCEPTION | MEMORY)  │
  └────────────────────────────────────────────────────────────────────────┘
```

</div>

---

## ⚡ Visual Overview — How Graphiti Solves Agent Amnesia

### 1. Vector RAG vs. Graphiti Temporal KG

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant VectorRAG as ❌ Vector RAG (Static Chunks)
    participant Graphiti as ✅ Graphiti KG (Bi-Temporal Facts)

    User->>VectorRAG: "Who is the current CTO of NovaTech?"
    VectorRAG-->>User: Returns 2018 chunk (Leon) & 2022 chunk (Aisha) → LLM Confused!

    User->>Graphiti: "Who is the current CTO of NovaTech?"
    Graphiti-->>User: Filters invalid_at != null → Returns "Aisha Okonkwo" (Valid Present)

    User->>Graphiti: "Who was CTO in June 2020?"
    Graphiti-->>User: Filters valid_at <= 2020-06-01 → Returns "Leon Müller" (Historical Fact)
```

---

### 2. Bi-Temporal Fact Lifecycle & Automatic Invalidation

```mermaid
gantt
    title NovaTech CTO Role Timeline (Bi-Temporal Invalidation)
    dateFormat  YYYY-MM-DD
    section Leon Müller (CTO)
    Active Role (valid_at: 2018-03-15)    :active, 2018-03-15, 2022-03-01
    Role Invalidated (invalid_at: 2022-03-01) :crit, 2022-03-01, 2026-09-18
    section Aisha Okonkwo (CTO)
    Appointed CTO (valid_at: 2022-03-01) :done, 2022-03-01, 2026-09-18
```

---

### 3. Visual Knowledge Graph State

```mermaid
graph TD
    subgraph Legend["Graph Legend"]
        L1["Active Fact (Present)"] --- L2["Invalidated Historical Fact"]
    end

    subgraph Timeline["NovaTech Temporal Knowledge Graph"]
        Org["🏢 NovaTech (Organization)"]

        Priya["👤 Priya Sharma (Person)"]
        Leon["👤 Leon Müller (Person)"]
        Aisha["👤 Aisha Okonkwo (Person)"]

        Priya -- "IS_CEO [2018-03-15 → Present]" --> Org
        Leon -- "WAS_CTO [2018-03-15 → 2022-03-01] ❌ INVALIDATED" --> Org
        Aisha -- "IS_CTO [2022-03-01 → Present] ✅ ACTIVE" --> Org
    end

    style Org fill:#161b22,stroke:#6C63FF,stroke-width:2px,color:#fff
    style Aisha fill:#00D4AA,stroke:#00D4AA,stroke-width:3px,color:#000
    style Leon fill:#222,stroke:#ff6b6b,stroke-dasharray: 5 5,color:#aaa
    style Priya fill:#008CC1,stroke:#008CC1,stroke-width:2px,color:#fff
    style L1 fill:#00D4AA,color:#000
    style L2 fill:#222,stroke:#ff6b6b,color:#aaa
```

---

## 🏗️ System Architecture (100% Local Zero-Cost Stack)

```mermaid
flowchart TB
    subgraph DataPerception["1. Perception Layer"]
        A[Raw Text Episodes / Chat Messages]
    end

    subgraph LocalModels["2. Local AI Engine (Ollama Docker)"]
        direction LR
        M1["phi3:mini (3.8B LLM)\nEntity & Relation Extraction"]
        M2["nomic-embed-text (768d)\nVector Similarity Search"]
    end

    subgraph GraphEngine["3. Graphiti Core Engine"]
        E1[Extraction & Claim Resolution]
        E2[Temporal Invalidation Engine]
        E3[Bi-Temporal Indexer]
    end

    subgraph Storage["4. Graph Database"]
        DB[(Neo4j 5.x Container\nbolt://localhost:7687)]
    end

    subgraph LangGraphAgent["5. LangGraph Stateful Memory Agent"]
        direction LR
        S1[retrieve_context_node] --> S2[generate_node] --> S3[save_to_graph_node]
    end

    A --> E1
    E1 <--> LocalModels
    E1 --> E2 --> E3 --> DB
    DB <--> LangGraphAgent

    style LocalModels fill:#111,stroke:#00D4AA,stroke-width:2px,color:#fff
    style Storage fill:#008CC1,stroke:#fff,stroke-width:2px,color:#fff
    style LangGraphAgent fill:#161b22,stroke:#6C63FF,stroke-width:2px,color:#fff
```

---

## ⚡ LangGraph Memory Agent Loop

```mermaid
stateDiagram-v2
    [*] --> RetrieveContextNode: User Prompt Received
    
    state RetrieveContextNode {
        [*] --> SearchGraphiti: Query Neo4j for valid facts
        SearchGraphiti --> RankTemporalFacts: Filter by valid_at & reference_time
    }
    
    RetrieveContextNode --> GenerateNode: Relevant Facts Grounding
    
    state GenerateNode {
        [*] --> SynthesizeAnswer: Local LLM generates grounded answer
    }
    
    GenerateNode --> SaveToGraphNode: Answer Generated
    
    state SaveToGraphNode {
        [*] --> IngestTurn: Ingest Q&A turn into Graphiti
        IngestTurn --> UpdateNeo4j: Create new nodes & update edges
    }
    
    SaveToGraphNode --> [*]: Return Answer to User
```

---

## 🚀 Quick Start (Zero-Cost Setup)

### 1. Launch Containers & Pull Local AI Models
```bash
# Start Neo4j & Ollama containers
docker compose up -d

# Download open-source models (phi3:mini LLM + nomic-embed-text embedder)
bash setup_models.sh
```

### 2. Set Up Python Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🧪 Tasks & Execution Guide

### Task 1 — Build Knowledge Graph
```bash
python 1_ingest_graph.py
```
> Extracts entities, temporal relations, and invalidation rules directly into Neo4j.

### Task 2 — Temporal Graph Queries
```bash
python 2_query_graph.py
```
> Demonstrates historical point-in-time search vs. present active state search.

### Task 3 — Stateful LangGraph Agent
```bash
python 3_langgraph_agent.py
```
> Agent automatically retrieves temporal graph context before generating responses and saves cross-session memory.

---

## 📊 Graphiti (Local Open Source) vs. Zep Cloud

| Parameter | 🏠 Self-Hosted Graphiti | ☁️ Zep Cloud |
|---|---|---|
| **Deployment** | Local Docker (Neo4j + Ollama) | Managed Serverless Cloud API |
| **API Cost** | **$0.00 (100% Free)** | Paid Tier / Consumption-based |
| **Data Privacy** | Air-gapped / Complete local isolation | SOC2 Compliant Cloud Storage |
| **Latency** | Dependent on CPU/GPU hardware | Optimized < 200ms Cloud SLA |
| **Setup Time** | ~2 minutes (`docker compose up`) | Instant API Key generation |
| **Infrastructure Control**| Full access to Cypher & Graph schemas | Managed abstract API endpoints |

---

## 🔍 Neo4j Visual Explorer

Inspect your generated graph directly in your browser:
- **URL:** [http://localhost:7474](http://localhost:7474)
- **Login:** `neo4j` / `graphiti123`

```cypher
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50;
```
