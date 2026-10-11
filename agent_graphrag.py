import os
import sys
import logging
import json
import re
import time
from typing import Dict, Any, List, Optional
import ollama

# Import the GraphRAG FastMCP functions
from mcp_graphrag import (
    vector_hybrid_search,
    graph_entity_search,
    graph_synthesis_query
)

# Logging setup
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter("%(message)s"))
root_logger.setLevel(logging.INFO)
root_logger.addHandler(console_handler)

logger = logging.getLogger("GraphRAG_Agent")


class MCPClientRegistry:
    """Direct call interface for GraphRAG MCP tools."""
    def __init__(self):
        self.tools = {
            "vector_hybrid_search": vector_hybrid_search,
            "graph_entity_search": graph_entity_search,
            "graph_synthesis_query": graph_synthesis_query
        }

    def execute_mcp_tool(self, tool_name: str, **kwargs) -> str:
        logger.info(f"[MCP CLIENT] Executing tool '{tool_name}' with args: {kwargs}")
        if tool_name in self.tools:
            try:
                return str(self.tools[tool_name](**kwargs))
            except Exception as e:
                logger.error(f"[MCP TOOL ERROR] {tool_name} failed: {str(e)}")
                return f"MCP Tool execution error: {str(e)}"
        return f"Tool '{tool_name}' not found."


class GraphRAGHelpDesk:
    def __init__(
        self, 
        reasoning_model: str = "gemma4", 
        fast_model: str = "gemma:2b"
    ):
        self.reasoning_model = reasoning_model
        self.fast_model = fast_model
        self.mcp_client = MCPClientRegistry()

    def _call_llm(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        image_path: Optional[str] = None, 
        tier: str = "reasoning"
    ) -> str:
        target_model = self.reasoning_model if (image_path or tier == "reasoning") else self.fast_model
        start_time = time.time()

        try:
            message_payload = {"role": "user", "content": f"System Task: {system_prompt}\nUser Query: {user_prompt}"}
            
            if image_path and os.path.exists(image_path):
                message_payload["images"] = [image_path]
    
            response = ollama.chat(                
                model=target_model,
                messages=[message_payload],
                options={"temperature": 0.0} # Enforce deterministic output

            )
            elapsed = time.time() - start_time
            content = response["message"]["content"].strip()
            logger.info(f"[ROUTER METRICS] Tier: '{tier}' | Model: '{target_model}' | Latency: {elapsed:.2f}s")
            return content

        except Exception as e:
            logger.error(f"Ollama execution error on model '{target_model}': {str(e)}")
            if tier == "reasoning":
                logger.warning(f"[FAILOVER TRIGGERED] Falling back to fast model '{self.fast_model}'...")
                return self._call_llm(system_prompt, user_prompt, image_path=None, tier="fast")
            return f"Error executing local model ({target_model}): {str(e)}"

    def agent_intent_router(self, query: str, image_path: Optional[str] = None) -> str:
        logger.info("-> AGENT 1 [Dynamic Router]: Evaluating GraphRAG intent...")

        system_prompt = (
            "You are an Intent Classification Agent for a Knowledge Graph System.\n"
            "Categorize the user query into EXACTLY ONE keyword:\n\n"
            "Examples:\n"
            "Query: 'How does X end up causing Y?' -> PATH_SYNTHESIS\n"
            "Query: 'What is the connection between ship traffic and sea creatures?' -> PATH_SYNTHESIS\n"
            "Query: 'What are the specs of Component A?' -> ENTITY_LOOKUP\n"
            "Query: 'Give me a general overview of marine biology.' -> VECTOR_SEARCH\n\n"
            "Categories: VECTOR_SEARCH, ENTITY_LOOKUP, PATH_SYNTHESIS, GENERAL\n"
            "Return ONLY the single category keyword."
        )


        intent = self._call_llm(system_prompt, query, tier="fast").upper()
        
        for valid in ["VECTOR_SEARCH", "ENTITY_LOOKUP", "PATH_SYNTHESIS", "GENERAL"]:
            if valid in intent:
                return valid
        return "GENERAL"

    def agent_vector_specialist(self, query: str) -> str:
        logger.info("-> AGENT 2 [Vector Specialist]: Generating query embedding & executing Hybrid Vector Search...")
        
        # Generate query vector embedding using nomic-embed-text
        try:
            emb_res = ollama.embeddings(model="nomic-embed-text", prompt=query)
            query_embedding = emb_res["embedding"]
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return f"Embedding generation error: {e}"

        mcp_data = self.mcp_client.execute_mcp_tool(
            "vector_hybrid_search", 
            query_embedding=query_embedding, 
            top_k=3
        )
        
        system_prompt = (
            "You are a Context Retrieval Specialist.\n"
            "Answer the user query strictly based on the following retrieved passages:\n"
            f"{mcp_data}"
        )
        return self._call_llm(system_prompt, query, tier="reasoning")

    def agent_entity_specialist(self, query: str) -> str:
        logger.info("-> AGENT 3 [Entity Specialist]: Extracting entity for Subgraph Retrieval...")
        
        # Extract target entity name via fast model
        extract_prompt = (
            "Extract the main entity/node name from the query.\n"
            "Respond ONLY in valid raw JSON format: {\"entity_name\": \"Entity\"}"
        )
        raw_output = self._call_llm(extract_prompt, query, tier="fast")
        
        entity = ""
        try:
            cleaned = raw_output.replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned)
            entity = data.get("entity_name", "").strip()
        except Exception:
            entity = re.sub(r'[^\w\s]', '', raw_output).strip()

        if not entity:
            entity = query

        logger.info(f"[EXTRACTED ENTITY]: '{entity}'")
        mcp_data = self.mcp_client.execute_mcp_tool("graph_entity_search", entity_name=entity, max_depth=2)

        system_prompt = (
            "You are a Knowledge Graph Entity Specialist.\n"
            "Synthesize a clear summary based ONLY on these graph connections:\n"
            f"{mcp_data}"
        )
        return self._call_llm(system_prompt, query, tier="reasoning")

    def agent_path_specialist(self, query: str) -> str:
        logger.info("-> AGENT 4 [Path Synthesis Specialist]: Extracting entity pair...")
        
        extract_prompt = (
            "Extract two target entities mentioned in the query.\n"
            "Respond ONLY in valid raw JSON format: {\"entity_a\": \"First\", \"entity_b\": \"Second\"}"
        )
        raw_output = self._call_llm(extract_prompt, query, tier="fast")
        
        entity_a, entity_b = "", ""
        try:
            cleaned = raw_output.replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned)
            entity_a = data.get("entity_a", "")
            entity_b = data.get("entity_b", "")
        except Exception:
            pass

        mcp_data = self.mcp_client.execute_mcp_tool("graph_synthesis_query", entity_a=entity_a, entity_b=entity_b)

        system_prompt = (
            "You are a Knowledge Graph Path Specialist.\n"
            "Explain the relationship network linking these entities using the provided graph path:\n"
            f"{mcp_data}"
        )
        return self._call_llm(system_prompt, query, tier="reasoning")

    def process_query(self, user_query: str, image_path: Optional[str] = None) -> str:
        logger.info("=" * 70)
        logger.info(f"USER QUERY: \"{user_query}\"")
        logger.info("=" * 70)

        intent = self.agent_intent_router(user_query, image_path=image_path)

        if intent == "VECTOR_SEARCH":
            return self.agent_vector_specialist(user_query)
        elif intent == "ENTITY_LOOKUP":
            return self.agent_entity_specialist(user_query)
        elif intent == "PATH_SYNTHESIS":
            return self.agent_path_specialist(user_query)
        else:
            return self._call_llm("You are an intelligent GraphRAG Assistant.", user_query, tier="fast")
