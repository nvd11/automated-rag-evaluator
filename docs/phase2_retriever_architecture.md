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
    class IQueryEmbedder {
        <<Interface>>
        +embed_query(text: String) List[float]
    }
    
    class IRetrieverDAO {
        <<Interface>>
        +semantic_search(query: SearchQuery) List[RetrievedContext]
    }
    
    class BaseRetriever {
        <<Interface>>
        +retrieve(text: String, top_k: int, filters: List) List[RetrievedContext]
    }

    %% Implementations (Concrete Layer)
    class GeminiQueryEmbedder {
        +embed_query(text: String) List[float]
    }
    
    class PgVectorRetrieverDAO {
        +semantic_search(query: SearchQuery) List[RetrievedContext]
    }
    
    class SemanticRetriever {
        -IQueryEmbedder embedder
        -IRetrieverDAO dao
        +retrieve(text: String, top_k: int, filters: List) List[RetrievedContext]
    }

    %% Relationships
    SearchQuery <-- IRetrieverDAO : Consumes
    RetrievedContext <-- IRetrieverDAO : Produces
    
    IQueryEmbedder <|.. GeminiQueryEmbedder : Implements
    IRetrieverDAO <|.. PgVectorRetrieverDAO : Implements
    BaseRetriever <|.. SemanticRetriever : Implements
    
    SemanticRetriever o-- IQueryEmbedder : Dependency Injection
    SemanticRetriever o-- IRetrieverDAO : Dependency Injection
```

---



## 2.5 Logic Flow Diagram (Sequence Diagram)

This sequence diagram illustrates the runtime behavior and interaction between the decoupled components during a retrieval request.

```mermaid
sequenceDiagram
    actor Client as Evaluator / User
    participant Orchestrator as SemanticRetriever
    participant Embedder as GeminiQueryEmbedder
    participant DAO as PgVectorRetrieverDAO
    participant DB as Cloud SQL (pgvector)

    Client->>Orchestrator: retrieve("What is HSBC's profit?", top_k=5, topics=["Financial Performance"])
    
    activate Orchestrator
    
    %% Step 1: Embedding
    Orchestrator->>Embedder: embed_query("What is HSBC's profit?")
    activate Embedder
    Note over Embedder: Calls Google AI Studio API
    Embedder-->>Orchestrator: return Vector [0.12, -0.45, ... 768d]
    deactivate Embedder
    
    %% Step 2: DTO Creation
    Note over Orchestrator: Creates SearchQuery DTO
    
    %% Step 3: Database Search
    Orchestrator->>DAO: semantic_search(SearchQuery)
    activate DAO
    
    Note over DAO: 1. Apply Metadata Pre-filter (Topic JOIN)<br/>2. Compute Cosine Distance (<=>)
    DAO->>DB: Execute SELECT with ORDER BY embedding <=> query_vector LIMIT 5
    activate DB
    DB-->>DAO: return Top 5 Rows (text, doc_id, page, score)
    deactivate DB
    
    Note over DAO: Maps Rows -> RetrievedContext DTOs
    DAO-->>Orchestrator: return List[RetrievedContext]
    deactivate DAO
    
    Orchestrator-->>Client: return Results
    deactivate Orchestrator
```

## 3. Core Components Deep Dive

### 3.1 Domain Models (DTOs)
Located in `src/domain/models.py`.
*   **`SearchQuery`**: Encapsulates the user's raw text, the generated embedding vector, the `top_k` parameter, and any optional metadata filters (e.g., `topic_filters = ["Risk Management"]`).
*   **`RetrievedContext`**: Represents the output of a search. Contains the chunk text, the parent `doc_id`, the origin page number, and critically, the `similarity_score` (calculated via cosine distance).

### 3.2 Interface Layer (Contracts)
Located in `src/interfaces/retriever_interfaces.py`.
By programming against interfaces rather than concrete classes, we follow the **Dependency Inversion Principle**.
*   **`IQueryEmbedder`**: Exposes a single method `embed_query()`. Unlike the batch embedder in Phase 1, this is optimized for low-latency, real-time single query embedding.
*   **`IRetrieverDAO`**: Defines the contract for vector databases. Accepts a `SearchQuery` and returns a list of `RetrievedContext`.

### 3.3 Concrete Implementations
Located in `src/retrieval/`.
*   **`GeminiQueryEmbedder`**: Implements `IQueryEmbedder` using the `google-genai` SDK. Translates the user's question into a 768-dimensional float array.
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
