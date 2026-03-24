import pytest

from src.diagnosis.rules import (
  RetrievalQualityRule,
  HallucinationRule,
  AnswerRelevanceRule,
  BenchmarkCorrectnessRule,
  DiagnosticEngine
)

class TestDiagnosticRules:
  """
  Unit tests ensuring the Heuristic Expert Rule Engine correctly triggers
  (or gracefully ignores) metrics based on strict scoring thresholds.
  """

  def test_retrieval_quality_rule_triggers_on_low_score(self):
    rule = RetrievalQualityRule(threshold=3.5)
    # Context score is 2.0 (Below 3.5)
    metrics = {"context_relevance": 2.0}
    
    diagnosis = rule.analyze(metrics)
    
    assert diagnosis is not None
    assert diagnosis.issue == "Low Retrieval Quality"
    assert len(diagnosis.likely_root_causes) > 0
    assert "hybrid_bm25_dense" in str(diagnosis.recommended_actions)

  def test_retrieval_quality_rule_ignores_high_score(self):
    rule = RetrievalQualityRule(threshold=3.5)
    # Context score is 4.0 (Above 3.5)
    metrics = {"context_relevance": 4.0}
    
    diagnosis = rule.analyze(metrics)
    assert diagnosis is None

  def test_hallucination_rule_triggers_on_low_faithfulness(self):
    rule = HallucinationRule(threshold=4.0)
    # Faithfulness is 3.1 (Below 4.0 = High Hallucination)
    metrics = {"faithfulness": 3.1}
    
    diagnosis = rule.analyze(metrics)
    
    assert diagnosis is not None
    assert diagnosis.issue == "High Generator Hallucination"
    assert "strict_citation_low_temp" in str(diagnosis.recommended_actions)

  def test_answer_relevance_rule_triggers_only_when_context_is_good(self):
    rule = AnswerRelevanceRule(context_threshold=3.5, answer_threshold=3.5)
    
    # Scenario 1: Good Context (4.0) but Bad Answer (2.0) -> Should Trigger!
    metrics_trigger = {"context_relevance": 4.0, "answer_relevance": 2.0}
    diagnosis = rule.analyze(metrics_trigger)
    assert diagnosis is not None
    assert "Poor Answer Relevance despite Good Retrieval" in diagnosis.issue
    assert "cross_encoder_miniLM" in str(diagnosis.recommended_actions)
    
    # Scenario 2: Bad Context (2.0) and Bad Answer (2.0) -> Should NOT Trigger 
    # (It's the retriever's fault, not the generator/reranker's fault)
    metrics_ignore_bad_ctx = {"context_relevance": 2.0, "answer_relevance": 2.0}
    assert rule.analyze(metrics_ignore_bad_ctx) is None
    
    # Scenario 3: Good Context (4.0) and Good Answer (4.5) -> Should NOT Trigger
    metrics_ignore_good_ans = {"context_relevance": 4.0, "answer_relevance": 4.5}
    assert rule.analyze(metrics_ignore_good_ans) is None

  def test_benchmark_correctness_rule_triggers_on_low_score(self):
    rule = BenchmarkCorrectnessRule(threshold=4.0)
    # Correctness is 3.5 (Below 4.0)
    metrics = {"correctness": 3.5}
    
    diagnosis = rule.analyze(metrics)
    
    assert diagnosis is not None
    assert "Low Accuracy against Benchmark" in diagnosis.issue
    assert "decompose_then_merge" in str(diagnosis.recommended_actions)

  def test_expert_engine_aggregation(self):
    """
    Tests the Orchestrator Engine's ability to chain rules together 
    and output a comprehensive list of diagnoses.
    """
    engine = DiagnosticEngine()
    engine.register_rule(RetrievalQualityRule(threshold=3.5))
    engine.register_rule(HallucinationRule(threshold=4.0))
    engine.register_rule(BenchmarkCorrectnessRule(threshold=4.0))
    
    # A terrible run: Bad retrieval, high hallucination, poor correctness
    metrics = {
      "context_relevance": 2.0,
      "faithfulness": 1.5,
      "correctness": 2.5
    }
    
    diagnoses = engine.diagnose(metrics)
    
    # All 3 rules should have fired
    assert len(diagnoses) == 3
    issues = [d.issue for d in diagnoses]
    assert "Low Retrieval Quality" in issues
    assert "High Generator Hallucination" in issues
    assert "Low Accuracy against Benchmark Ground Truth" in issues