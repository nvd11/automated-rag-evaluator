# Phase 5: Evaluation Runner Architecture
*LLM-as-a-Judge and The "Upgraded EAV" Metrics Pipeline*

## 1. Overview
The final phase of our RAG testing framework is **Phase 5: The Evaluation Runner**. While Phase 4 generated the answers, Phase 5 is responsible for grading them. This component operates entirely asynchronously from the generation phase, meaning it can load *historical* inference runs from the database and evaluate them long after the RAG Agent has finished its job.

The Evaluation Runner relies on an advanced LLM ("The Judge") to analyze the generated answers. Crucially, it must intelligently support our dual-track requirements:
- **Case 1 (Benchmark Scoring):** If Ground Truth exists, it measures Answer Correctness and Semantic Similarity.
- **Case 2 (The RAG Triad):** If no Ground Truth exists (e.g., live production logs), it evaluates Context Relevance, Faithfulness (Hallucination), and Answer Relevance.

## 2. Architectural Intent: The Upgraded EAV Model
In early prototypes, evaluation metrics were often pivoted into wide tables or strictly normalized into Entity-Attribute-Value (EAV) schemas. However, simple EAV schemas fail to capture the most valuable output of LLM-as-a-Judge systems: **Explainability**.

Our architecture utilizes an **Upgraded EAV Model 2.0** for the `evaluation_metrics` table:
1. **Per-Metric Granularity:** Each individual score (e.g., *Faithfulness*) gets its own row in the database, linked back to the `query_id`.
2. **Explicit Reasoning:** Alongside the numerical `metric_value` (e.g., 4.5), we store the LLM's `reasoning` (TEXT) explaining exactly *why* it awarded that score.
3. **Evaluation Strategy Isolation:** The `evaluation_strategy` field (e.g., 'CASE1_GROUND_TRUTH', 'CASE2_RAG_TRIAD') combined with the `judge_model` name ensures that a single query can be evaluated multiple times using different methodologies without metric collision.

This design guarantees infinite extensibility for future metrics while preserving exact lineage and explainability for BI dashboard audits.

## 3. Entity-Relationship (ER) Diagram

The following UML focuses specifically on how the Evaluation components interact with the operational logs. Notice the absence of cascading deletes (Hard FKs) on the `evaluation_metrics` table, ensuring historical scores survive even if the raw logs are purged.

```mermaid
erDiagram
  %% Source Operational Logs
  query_history {
    UUID query_id PK
    TEXT question
    TEXT generated_answer
    JSONB retrieved_contexts
  }
  
  golden_records {
    UUID id PK
    TEXT ground_truth
  }
  
  golden_record_query_mapping {
    UUID query_id
    UUID golden_record_id
  }

  %% Evaluation Storage (Upgraded EAV)
  evaluation_metrics {
    UUID id PK
    UUID query_id "SOFT FK"
    VARCHAR evaluation_strategy
    VARCHAR metric_category
    VARCHAR metric_name
    NUMERIC metric_value
    TEXT reasoning
    VARCHAR judge_model
  }

  %% Relationships
  query_history ||--o| golden_record_query_mapping : "Joined via"
  golden_records ||--o| golden_record_query_mapping : "Resolves GT"
  query_history ||--o{ evaluation_metrics : "Evaluated by (Soft Link)"
```

## 4. Execution Flow (Sequence Diagram)

The `EvaluationRunner` acts simply as the CLI/cron entrypoint. It configures the dependencies and delegates all heavy lifting to the `EvaluationPipeline`. The Pipeline orchestrates the extraction of historical inference data, automatically categorizing queries via the `QueryEvaluationDTO.has_ground_truth` property, and dispatches them to the appropriate polymorphic `ILLMJudge` implementation.

```mermaid
sequenceDiagram
  actor DataScientist
  participant Runner as EvaluationRunner
  participant Pipeline as EvaluationPipeline
  participant DAO as EvaluationDAO
  participant JudgeG as GoldenBaselineJudge (LLM)
  participant JudgeT as RagTriadJudge (LLM)
  participant DB as Cloud SQL (PostgreSQL)

  DataScientist->>Runner: main()
  activate Runner
  
  Runner->>Pipeline: run(inference_run_id="123-abc")
  activate Pipeline
  
  Note over Pipeline, DB: Step 1: Load Data & Detect Strategy
  Pipeline->>DAO: fetch_queries_for_evaluation(run_id)
  
  %% The DAO automatically LEFT JOINs with golden_record_query_mapping
  DAO-->>Pipeline: Return List[QueryEvaluationDTO]
  
  Note over Pipeline, JudgeT: Step 2: Polymorphic Concurrency
  loop For each Query (asyncio.gather / Semaphore)
    alt has_ground_truth == True (Case 1)
      Pipeline->>JudgeG: evaluate_query(dto)
      activate JudgeG
      Note over JudgeG: Prompt: Compare Answer to GT
      JudgeG-->>Pipeline: Return List[ScoreWithReasoning] (Correctness)
      deactivate JudgeG
      
    else has_ground_truth == False (Case 2)
      Pipeline->>JudgeT: evaluate_query(dto)
      activate JudgeT
      Note over JudgeT: Prompt: Assess RAG Triad (Faithfulness, etc.)
      JudgeT-->>Pipeline: Return List[ScoreWithReasoning] (Triad)
      deactivate JudgeT
    end
  end
  
  Note over Pipeline, DB: Step 3: Transactional Bulk Persistence
  Pipeline->>DAO: bulk_insert_evaluation_metrics(metrics)
  DAO->>DB: INSERT INTO evaluation_metrics
  
  Pipeline-->>Runner: Evaluation Summary / Status
  deactivate Pipeline
  Runner-->>DataScientist: Run Complete
  deactivate Runner
```

## 5. Architectural Intent: Polymorphic Evaluators (Open-Closed Principle)

To ensure the framework is very extensible, the evaluation logic is decoupled using pure object-oriented polymorphism. The base interface `ILLMJudge` defines a single contract:
```python
@abstractmethod
async def evaluate_query(self, dto: QueryEvaluationDTO) -> List[ScoreWithReasoning]:
  pass
```

Instead of hardcoding case-specific methods (e.g., `evaluate_case1`, `evaluate_case2`), we inject distinct concrete implementations into the pipeline:
- **`GoldenBaselineJudge`**: Implements `evaluate_query` by strictly comparing the generated answer against `dto.ground_truth`.
- **`RagTriadJudge`**: Implements `evaluate_query` by scoring the RAG Triad (Faithfulness, Answer Relevance, Context Relevance) against `dto.retrieved_contexts`.

This rigorously adheres to the Open-Closed Principle (OCP)—if a new evaluation strategy (e.g., "Case 3: Heuristic Keyword Matcher") is required tomorrow, developers simply create a new judge class extending `ILLMJudge` without modifying the core pipeline or interface.

### Dependency Injection and Factory Reuse
Crucially, these concrete judge implementations **do not** instantiate their own LLM clients. Instead, they rely on Dependency Injection (DI) through their constructors, accepting a generic `BaseChatModel`. 

This design explicitly reuses our existing `GeminiLLMFactory` (which already encapsulates complex logic like proxy fallback configurations, timeout handling, and Google SDK initialization). 
- **Single Responsibility Principle (SRP):** The judges only focus on "How to prompt for scores," while the factory manages "How to connect to the model."
- **Testability:** Mocking the LLM for unit tests becomes trivial by passing a `FakeListChatModel` into the constructor.
- **Provider Agnosticism:** Switching the judge from Gemini to OpenAI's GPT-4 requires exactly zero code changes inside the judge classes; the runner simply injects a different factory output.

## 6. Structured LLM Output Design (LangChain Function Calling)

To guarantee the `LLMJudge` returns stable, parseable data capable of populating our EAV table, we utilize LangChain's `.with_structured_output()`. The LLM is forced to return a JSON array matching the following Pydantic schema:

```python
class ScoreWithReasoning(BaseModel):
  metric_name: str = Field(description="Name of the metric (e.g., 'Faithfulness')")
  score: float = Field(description="Numerical score from 0.0 to 5.0")
  reasoning: str = Field(description="Detailed explanation justifying the score")
```

This guarantees that the Runner never encounters string parsing errors (like `RegexParserError`) and can instantly map the output directly to the database columns.