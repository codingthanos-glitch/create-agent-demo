# LangChain Agent Loop Demo

A hands-on implementation of a basic LLM agent loop using LangChain, Qwen3, Ollama, and a custom Python tool.

Instead of using LangChain's high-level `create_agent()` API, this project manually implements the fundamental agent loop to demonstrate how an LLM interacts with tools.

## Overview

An LLM agent is essentially a loop in which the model can:

1. Receive a user request.
2. Decide whether a tool is required.
3. Request a tool call.
4. Let the application execute the tool.
5. Receive the tool result.
6. Continue reasoning with the updated conversation.
7. Produce a final response when no more tools are required.

The core flow is:

```text
User
  |
  v
HumanMessage
  |
  v
Qwen3
  |
  | tool call
  v
AIMessage
  |
  v
Python Tool
  |
  v
ToolMessage
  |
  v
Qwen3
  |
  | no more tools
  v
Final AIMessage
