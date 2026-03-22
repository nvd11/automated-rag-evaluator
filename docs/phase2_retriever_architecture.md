# Phase 2: RAG Retriever Architecture Design

## 1. Overview
The Retriever module is the critical bridge between the user's query and the LLM's generation/evaluation phase. Its sole responsibility is to accurately and efficiently fetch the most semantically relevant context chunks from the database.

To demonstrate senior-level Object-Oriented Programming (OOP) and strictly adhere to **SOLID principles**, the architecture isolates concerns into distinct layers: Data Transfer Objects (DTOs), Interfaces (Contracts), Concrete Implementations, and an Orchestrator. This ensures the Retriever is perfectly decoupled and highly testable.

---

## 2. Architecture Diagram (Mermaid)

```mermaid
classDiagram
    %% Data Transfer Objects (Domain Layer)
    class SearchQuery {
        +String query_text
        +List[float] embedding
        +int top_k
        +List[str] topic_filters
    }
    
    class RetrievedContext {
        +String doc_id
        +String chunk_id
        +String text
        +float similarity_score
        +dict metadata
    }

    %% Interfaces (Contract Layer)
    %% Reusing BaseEmbedder from Ingestion
    class BaseEmbedder {
        <<Interface>>
        +embed_query(text: String) List[float]
    }
    
    class IRetrieverDAO {
        <<Interface>>
        +semantic_search(query: SearchQuery) List[RetrievedContext]
    }
    
    class ILLMGenerator {
        <<Interface>>
        +generate_answer(prompt: String) String
    }
    
    class BaseRetriever {
        <<Interface>>
        +retrieve(text: String, top_k: int, filters: List) List[RetrievedContext]
    }

    %% Implementations (Concrete Layer)
    class GeminiEmbedder {
        +embed_query(text: String) List[float]
    }
    
    class LangchainGeminiGenerator {
        +generate_answer(prompt: String) String
        %% Note: Uses LCEL (LangChain Expression Language)
    }
    
    class PgVectorRetrieverDAO {
        +semantic_search(query: SearchQuery) List[RetrievedContext]
    }
    
    class SemanticRetriever {
        -BaseEmbedder embedder
        -IRetrieverDAO dao
        +retrieve(text: String, top_k: int, filters: List) List[RetrievedContext]
    }

    class RAGResponse {
        +String query
        +String generated_answer
        +List[RetrievedContext] retrieved_contexts
    }

    class RAGAgent {
        -BaseRetriever retriever
        -ILLMGenerator generator
        +ask(question: String, top_k: int, topics: List) RAGResponse
    }

    RAGAgent o-- BaseRetriever : Dependency Injection
    RAGAgent o-- ILLMGenerator : Dependency Injection
    
    %% Relationships
    SearchQuery <-- IRetrieverDAO : Consumes
    RetrievedContext <-- IRetrieverDAO : Produces
    
    BaseEmbedder <|.. GeminiEmbedder : Implements
    IRetrieverDAO <|.. PgVectorRetrieverDAO : Implements
    ILLMGenerator <|.. LangchainGeminiGenerator : Implements
    BaseRetriever <|.. SemanticRetriever : Implements
    
    SemanticRetriever o-- BaseEmbedder : Dependency Injection
    SemanticRetriever o-- IRetrieverDAO : Dependency Injection
```

---




## 2.5 Logic Flow Diagram (End-to-End RAG Sequence)

This sequence diagram illustrates the runtime behavior of the Retriever and how it integrates into the broader Retrieval-Augmented Generation (RAG) pipeline to synthesize a final answer.

```mermaid
sequenceDiagram
    actor Client as User / Evaluator
    participant RAG as RAGAgent (Orchestrator)
    participant Retriever as SemanticRetriever
    participant Embedder as GeminiEmbedder
    participant DAO as PgVectorRetrieverDAO
    participant DB as Cloud SQL (pgvector)
    participant LLM as LangchainGeminiGenerator (LCEL)

    Client->>RAG: ask("What is HSBC's profit?", topics=["Financial Performance"])
    activate RAG

    %% PHASE 1: Retrieval
    Note over RAG, DB: PHASE 1: Semantic Retrieval (The "R" in RAG)
    RAG->>Retriever: retrieve("What is HSBC's profit?", top_k=5, topics=[...])
    activate Retriever
    
    Retriever->>Embedder: embed_query("What is HSBC's profit?")
    activate Embedder
    Embedder-->>Retriever: return Vector [0.12, -0.45, ... 768d]
    deactivate Embedder
    
    Note over Retriever: Creates SearchQuery DTO
    Retriever->>DAO: semantic_search(SearchQuery)
    activate DAO
    
    Note over DAO: 1. Apply Metadata Pre-filter (Topic JOIN)<br/>2. Compute Cosine Distance (<=>)
    DAO->>DB: Execute SELECT with ORDER BY embedding <=> query_vector LIMIT 5
    activate DB
    DB-->>DAO: return Top 5 Rows (text, doc_id, page, score)
    deactivate DB
    
    DAO-->>Retriever: return List[RetrievedContext]
    deactivate DAO
    
    Retriever-->>RAG: return List[RetrievedContext]
    deactivate Retriever

    %% PHASE 2: Generation
    Note over RAG, LLM: PHASE 2: Augmented Generation (The "A & G" in RAG)
    Note over RAG: Constructs Prompt:<br/>"Based on {Context}, answer {Query}"
    
    RAG->>LLM: generate_content(Prompt)
    activate LLM
    LLM-->>RAG: return Synthesized Final Answer
    deactivate LLM
    
    Note over RAG: Packages Answer and List[RetrievedContext] into RAGResponse DTO
    RAG-->>Client: return RAGResponse
    deactivate RAG
```

## 3. Core Components Deep Dive

### 3.1 Domain Models (DTOs)
Located in `src/domain/models.py`.
*   **`SearchQuery`**: Encapsulates the user's raw text, the generated embedding vector, the `top_k` parameter, and any optional metadata filters (e.g., `topic_filters = ["Risk Management"]`).
*   **`RAGResponse`**: The ultimate output payload. By returning a structured object containing both the `generated_answer` and the full array of `retrieved_contexts` (including their raw `text` and `similarity_score`), this design satisfies both user-facing citation requirements and downstream machine-driven RAG evaluation (e.g., calculating Faithfulness).
*   **`RetrievedContext`**: Represents the output of a search. Contains the chunk text, the parent `doc_id`, the origin page number, and critically, the `similarity_score` (calculated via cosine distance).

### 3.2 Interface Layer (Contracts)
Located in `src/interfaces/retriever_interfaces.py`.
By programming against interfaces rather than concrete classes, we follow the **Dependency Inversion Principle**.
*   **`BaseEmbedder` (Reused from Phase 1)**: By extending our existing interface with `embed_query()`, we unify the embedding model for both ingestion and retrieval, ensuring vector space alignment.
*   **`IRetrieverDAO`**: Defines the contract for vector databases. Accepts a `SearchQuery` and returns a list of `RetrievedContext`.

### 3.3 Concrete Implementations
Located in `src/retrieval/`.
*   **`GeminiEmbedder`**: We reuse the implementation from Phase 1, adding a new `embed_query()` method explicitly configured with `task_type="RETRIEVAL_QUERY"` to match the documents embedded with `RETRIEVAL_DOCUMENT`.
*   **`PgVectorRetrieverDAO`**: Implements `IRetrieverDAO` using `psycopg` and `pgvector`. 
    *   **Query Strategy**: Utilizes the `<=>` operator for Cosine Distance.
    *   **Pre-filtering**: Dynamically constructs SQL `WHERE` clauses if `topic_filters` are provided. By joining `document_chunks` with `document_topics`, we narrow the vector search space *before* computing distances, drastically improving precision and query speed.

### 3.4 Orchestration
*   **`SemanticRetriever`**: The master class combining the components.
    *   **Constructor Injection**: Receives `embedder: IQueryEmbedder` and `dao: IRetrieverDAO`.
    *   **Execution Flow**: 
        1. Validates input.
        2. Calls `embedder.embed_query()` to get the vector.
        3. Constructs a `SearchQuery` DTO.
        4. Calls `dao.semantic_search()`.
        5. Returns the structured list of `RetrievedContext`.

---

## 4. Engineering Highlights (For the Interviewer)
1.  **Open/Closed Principle**: If we want to switch from PostgreSQL to Pinecone, or from Gemini to OpenAI embeddings, we simply create a new implementation class. The `SemanticRetriever` orchestrator requires zero modifications.
2.  **Testability**: Because of Dependency Injection, we can trivially unit test the `SemanticRetriever` by injecting a `MockQueryEmbedder` and a `MockRetrieverDAO` without needing an active database connection or spending API credits.
3.  **Hybrid Search Readiness**: The `SearchQuery` DTO and the `IRetrieverDAO` interface are designed so that future upgrades (like adding keyword-based BM25 search) can be seamlessly integrated.
