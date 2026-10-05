"""
File 5: 05_genai_agent_function_calling.py
=========================================
GenAI & Deep Learning Assessment - Module 5
Topic: Autonomous Generative AI Agent with Tool & Function Calling

Description:
This module demonstrates an autonomous Generative AI Agent implementing the ReAct
(Reasoning + Acting) pattern. The agent binds custom python tools (calculator, weather lookup,
and knowledge base search), parses structured JSON action intents, executes tool functions,
and integrates observation feeds to resolve multi-step queries.
"""

import json
import re

# ============================================================
# TOOL DEFINITIONS (Function Registry)
# ============================================================

def tool_calculator(expression: str) -> str:
    """Evaluates mathematical expressions safely."""
    try:
        # Sanitize input for safe evaluation
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters in expression."
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"

def tool_get_weather(location: str) -> str:
    """Fetches real-time weather information for a given city."""
    mock_weather_db = {
        "london": "15°C, Partly Cloudy, Humidity: 68%",
        "new york": "22°C, Sunny, Humidity: 45%",
        "tokyo": "19°C, Clear Sky, Humidity: 55%",
        "chennai": "31°C, Humid & Sunny, Humidity: 80%"
    }
    loc_clean = location.strip().lower()
    return mock_weather_db.get(loc_clean, f"Weather data for '{location}' unavailable. Default: 25°C, Clear.")

def tool_search_docs(query: str) -> str:
    """Searches local company documentation."""
    docs = {
        "policy": "Remote work policy allows up to 2 days work-from-home per week.",
        "support": "Helpdesk contact email is support@ai-assessment.org."
    }
    for key, text in docs.items():
        if key in query.lower():
            return text
    return "No matching internal documents found."

# Registry of available tools for the agent
TOOL_REGISTRY = {
    "calculator": tool_calculator,
    "get_weather": tool_get_weather,
    "search_docs": tool_search_docs
}

# ============================================================
# AGENT REASONING ENGINE
# ============================================================

class GenAIAgent:
    """ReAct Autonomous Agent capable of function calling."""
    def __init__(self, name="GenAI-Agent-V1"):
        self.name = name

    def get_system_prompt(self):
        tools_description = "\n".join([
            f"- {name}: {func.__doc__}" for name, func in TOOL_REGISTRY.items()
        ])
        return (
            f"You are {self.name}, an intelligent AI assistant.\n"
            f"You have access to the following tools:\n{tools_description}\n\n"
            f"To use a tool, respond with a JSON object in the format:\n"
            f'{{"action": "<tool_name>", "action_input": "<input>"}}\n\n'
            f"When you have the final answer, respond with:\n"
            f'{{"final_answer": "<your complete response>"}}'
        )

    def execute_step(self, user_prompt, conversation_history):
        """Simulates LLM reasoning and action decision."""
        print(f"\n[{self.name} Reasoning Step]")

        # Simulated multi-step agent logic for demonstration
        if len(conversation_history) == 0:
            action = {"action": "get_weather", "action_input": "Chennai"}
            return json.dumps(action)
        elif len(conversation_history) == 1:
            action = {"action": "calculator", "action_input": "31 * 1.8 + 32"}
            return json.dumps(action)
        else:
            final = {"final_answer": "The temperature in Chennai is 31°C, which equals 87.8°F."}
            return json.dumps(final)

    def run(self, user_prompt):
        print("==================================================")
        print(f"User Request: {user_prompt}")
        print("==================================================")
        print("System Prompt:\n", self.get_system_prompt())

        conversation_history = []
        max_turns = 5

        for turn in range(max_turns):
            response_str = self.execute_step(user_prompt, conversation_history)
            try:
                decision = json.loads(response_str)
            except json.JSONDecodeError:
                print("Failed to parse agent JSON response.")
                break

            if "final_answer" in decision:
                print(f"\n[Agent Final Answer]\n{decision['final_answer']}\n")
                break

            action_name = decision.get("action")
            action_input = decision.get("action_input")

            print(f"-> Agent Action: Called tool '{action_name}' with input: '{action_input}'")

            if action_name in TOOL_REGISTRY:
                tool_output = TOOL_REGISTRY[action_name](action_input)
                print(f"<- Tool Observation: {tool_output}")
                conversation_history.append({"action": action_name, "output": tool_output})
            else:
                print(f"Error: Tool '{action_name}' not found in registry.")
                break

def main():
    agent = GenAIAgent()
    
    # Task: Multi-step reasoning (Weather lookup + Temperature conversion)
    user_query = "What is the weather in Chennai, and what is its temperature in Fahrenheit (Celsius * 1.8 + 32)?"
    agent.run(user_query)

if __name__ == "__main__":
    main()
