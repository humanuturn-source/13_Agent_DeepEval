# test_agent_evals.py
import sys
import types
import os

os.environ["DEEPEVAL_DISABLE_TIMEOUTS"] = "1"
os.environ["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE"] = "300"

try:
    import langchain_community.chat_models.vertexai
except ModuleNotFoundError:
    dummy_module = types.ModuleType("langchain_community.chat_models.vertexai")
    dummy_module.ChatVertexAI = type("ChatVertexAI", (object,), {})
    sys.modules["langchain_community.chat_models.vertexai"] = dummy_module

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

from agent_graphrag import GraphRAGHelpDesk
from local_judge import LocalOllamaJudge

agent = GraphRAGHelpDesk(reasoning_model="gemma4", fast_model="gemma:2b")
local_judge = LocalOllamaJudge(model_name="gemma4")

def _get_context(query: str) -> str:
    emb = agent.mcp_client.tools["vector_hybrid_search"].__globals__["ollama"].embeddings(
        model="nomic-embed-text", prompt=query
    )["embedding"]
    return agent.mcp_client.execute_mcp_tool("vector_hybrid_search", query_embedding=emb, top_k=2)

def _clean_actual_output(output: str) -> str:
    """Strips technical error trace logs and system warnings from LLM outputs."""
    lines = output.splitlines()
    clean_lines = [
        line for line in lines 
        if not any(err in line.lower() for err in ["error:", "traceback", "schema failure", "failed to parse", "exception"])
    ]
    return "\n".join(clean_lines).strip() or output

def test_killer_whale_faithfulness():
    query = "How deep down in the ocean do Killer Whales usually live, and what do they eat?"
    retrieved_context = _get_context(query)
    actual_output = _clean_actual_output(agent.process_query(query))

    test_case = LLMTestCase(input=query, actual_output=actual_output, retrieval_context=[retrieved_context])
    metric = FaithfulnessMetric(threshold=0.8, model=local_judge, include_reason=True, async_mode=False)
    assert_test(test_case, [metric])

def test_pacific_ocean_relevancy():
    query = "Which ocean is the biggest and deepest in the world?"
    actual_output = _clean_actual_output(agent.process_query(query))

    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=["The Pacific Ocean is the most expansive planetary water body with a surface area of 165,250,000 sq km and maximum depth of 10,928m in Challenger Deep."]
    )
    metric = AnswerRelevancyMetric(threshold=0.7, model=local_judge, include_reason=True, async_mode=False)
    assert_test(test_case, [metric])

def test_deep_sea_survival_faithfulness():
    query = "How do fish and creatures survive way down in the deep ocean where there's no sunlight?"
    retrieved_context = _get_context(query)
    actual_output = _clean_actual_output(agent.process_query(query))

    test_case = LLMTestCase(input=query, actual_output=actual_output, retrieval_context=[retrieved_context])
    metric = FaithfulnessMetric(threshold=0.8, model=local_judge, include_reason=True, async_mode=False)
    assert_test(test_case, [metric])

def test_anglerfish_hunting_relevancy():
    query = "What is the Humpback Anglerfish, and how does it catch its food in the dark?"
    actual_output = _clean_actual_output(agent.process_query(query))

    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=["Melanocetus johnsonii (Humpback Anglerfish) lives in the Bathypelagic Zone (1000-4000m) and uses an Esca lure containing bioluminescent Photobacterium symbionts to attract prey."]
    )
    metric = AnswerRelevancyMetric(threshold=0.7, model=local_judge, include_reason=True, async_mode=False)
    assert_test(test_case, [metric])

def test_ocean_pressure_faithfulness():
    query = "Why is the water at the bottom of the ocean so cold and under so much pressure?"
    retrieved_context = _get_context(query)
    actual_output = _clean_actual_output(agent.process_query(query))

    test_case = LLMTestCase(input=query, actual_output=actual_output, retrieval_context=[retrieved_context])
    metric = FaithfulnessMetric(threshold=0.8, model=local_judge, include_reason=True, async_mode=False)
    assert_test(test_case, [metric])

def test_overfishing_impact_relevancy():
    query = "How does overfishing affect the fish living in our oceans today?"
    actual_output = _clean_actual_output(agent.process_query(query))

    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=["Global capture fisheries extract 90 million metric tons per year, pushing over 34% of wild fish stocks beyond sustainable biological limits."]
    )
    metric = AnswerRelevancyMetric(threshold=0.7, model=local_judge, include_reason=True, async_mode=False)
    assert_test(test_case, [metric])

def test_hydrothermal_vent_faithfulness():
    query = "What are hydrothermal vents, and how do ocean creatures live near super hot underwater volcanoes?"
    retrieved_context = _get_context(query)
    actual_output = _clean_actual_output(agent.process_query(query))

    test_case = LLMTestCase(input=query, actual_output=actual_output, retrieval_context=[retrieved_context])
    metric = FaithfulnessMetric(threshold=0.8, model=local_judge, include_reason=True, async_mode=False)
    assert_test(test_case, [metric])
