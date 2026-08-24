from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage


# ----------------------------------------------------------------------
# 1. Create the LLM
# ----------------------------------------------------------------------
#
# ChatOllama provides LangChain's interface for communicating with
# models running locally through Ollama.
#
# Here we are using the Qwen3 8B model.
#
# temperature=0 makes the model's output more deterministic.
#
llm = ChatOllama(
    model="qwen3:8b",
    temperature=0
)


# ----------------------------------------------------------------------
# 2. Define a tool
# ----------------------------------------------------------------------
#
# @tool converts the normal Python function into a LangChain Tool.
#
# This gives LangChain information about:
#
#   - The tool name
#   - The tool description
#   - The function arguments
#   - The argument types
#
# The LLM will receive this tool information when we use bind_tools().
#
@tool
def get_ceo(company: str) -> str:
    """Return the CEO of a company from the local company database.

    The returned value is the authoritative answer for this application.
    Do not replace or contradict the returned CEO using outside knowledge.

    Args:
        company: Name of the company whose CEO is required.

    Returns:
        The CEO's name if the company exists in the local database.
        Otherwise, returns "CEO not found".
    """

    # Simple in-memory database containing company → CEO mappings.
    #
    # In a real application, this could instead be:
    #
    #   - A database query
    #   - A REST API call
    #   - A web search
    #   - Another MCP tool
    #
    ceo_database = {
        "google": "Sundar Pichai",
        "microsoft": "Satya Nadella",
        "apple": "Tim Cook",
        "meta": "Mark Zuckerberg"
    }

    # Normalize the company name to lowercase before looking it up.
    #
    # This allows:
    #
    #   Google
    #   google
    #   GOOGLE
    #
    # to all match the "google" key in the dictionary.
    #
    return ceo_database.get(
        company.lower(),
        "CEO not found"
    )


# ----------------------------------------------------------------------
# 3. Bind the tool to the LLM
# ----------------------------------------------------------------------
#
# bind_tools() tells the LLM which tools are available to it.
#
# IMPORTANT:
#
# This does NOT execute get_ceo().
#
# It only makes the tool available to the model so that the model
# can REQUEST that the tool be executed.
#
# The LLM can now produce a tool call such as:
#
#   get_ceo(company="Google")
#
# Our Python program is still responsible for actually executing
# the function.
#
llm = llm.bind_tools([get_ceo])


# ----------------------------------------------------------------------
# 4. Create the initial user query
# ----------------------------------------------------------------------
#
# This is the question that starts the agent loop.
#
query = "who is the CEO of Google?"


# ----------------------------------------------------------------------
# 5. Create the initial conversation state
# ----------------------------------------------------------------------
#
# messages stores the complete conversation history.
#
# We start with one HumanMessage representing the user's question.
#
# The message history will later become:
#
#   HumanMessage
#       ↓
#   AIMessage (tool call)
#       ↓
#   ToolMessage (tool result)
#       ↓
#   AIMessage (final answer)
#
messages = [
    HumanMessage(content=query)
]


# ----------------------------------------------------------------------
# 6. Start the agent loop
# ----------------------------------------------------------------------
#
# The loop continues until the LLM produces a response that does not
# contain a tool call.
#
# Each iteration performs the following:
#
#   1. Send the conversation history to the LLM.
#   2. Check whether the LLM requested a tool.
#   3. If yes, execute the requested tool.
#   4. Add the tool result to the conversation.
#   5. Go back to the LLM.
#
# When the LLM no longer requests a tool, its response is treated
# as the final answer and the loop ends.
#
while True:

    # ------------------------------------------------------------------
    # 6.1 Ask the LLM what to do next
    # ------------------------------------------------------------------
    #
    # We send the COMPLETE conversation history to the LLM.
    #
    # On the first iteration, this contains only the HumanMessage.
    #
    # After a tool call, it will also contain:
    #
    #   - The previous AIMessage
    #   - The ToolMessage containing the tool result
    #
    llm_response = llm.invoke(messages)


    # ------------------------------------------------------------------
    # 6.2 Store the LLM response in the conversation history
    # ------------------------------------------------------------------
    #
    # The LLM response is an AIMessage.
    #
    # If the LLM requested a tool, this AIMessage contains tool_calls.
    #
    # For example:
    #
    #   AIMessage(
    #       content="",
    #       tool_calls=[
    #           {
    #               "name": "get_ceo",
    #               "args": {"company": "Google"},
    #               "id": "abc123"
    #           }
    #       ]
    #   )
    #
    messages.append(llm_response)


    # ------------------------------------------------------------------
    # 6.3 Check whether the LLM requested a tool
    # ------------------------------------------------------------------
    #
    # tool_calls contains the tools requested by the LLM.
    #
    # If it is non-empty:
    #
    #     The LLM wants our application to execute a tool.
    #
    # If it is empty:
    #
    #     The LLM has produced its final response.
    #
    if llm_response.tool_calls:

        # --------------------------------------------------------------
        # 6.4 Get the first requested tool call
        # --------------------------------------------------------------
        #
        # For this learning example, we handle only one tool call.
        #
        # A production agent may need to handle multiple tool calls.
        #
        tool_call = llm_response.tool_calls[0]


        # --------------------------------------------------------------
        # 6.5 Extract information from the tool call
        # --------------------------------------------------------------
        #
        # tool_name:
        #     Name of the tool the LLM wants to execute.
        #
        # tool_args:
        #     Arguments generated by the LLM for that tool.
        #
        # tool_call_id:
        #     Unique ID identifying this particular tool call.
        #
        # The ID is later stored in ToolMessage so the tool result
        # can be associated with the correct tool call.
        #
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]


        # --------------------------------------------------------------
        # 6.6 Execute the requested tool
        # --------------------------------------------------------------
        #
        # The LLM does NOT execute Python code.
        #
        # The LLM only requested:
        #
        #     get_ceo(company="Google")
        #
        # Our Python program now executes the actual tool.
        #
        if tool_name == "get_ceo":

            tool_response = get_ceo.invoke(tool_args)

        else:

            # If the LLM requests a tool that our application does
            # not recognize, stop execution rather than silently
            # doing something unexpected.
            #
            raise ValueError(
                f"Unknown tool: {tool_name}"
            )


        # --------------------------------------------------------------
        # 6.7 Add the tool result to the conversation
        # --------------------------------------------------------------
        #
        # The tool result should be represented as a ToolMessage.
        #
        # The tool_call_id connects this result to the AIMessage
        # that requested the tool.
        #
        # For example:
        #
        #   AIMessage
        #       tool_call_id = "abc123"
        #
        #              ↓
        #
        #   ToolMessage
        #       tool_call_id = "abc123"
        #       content = "Sundar Pichai"
        #
        # The updated conversation can now be sent back to the LLM.
        #
        messages.append(
            ToolMessage(
                content=tool_response,
                tool_call_id=tool_call_id
            )
        )


    # ------------------------------------------------------------------
    # 6.8 No tool call → final answer
    # ------------------------------------------------------------------
    #
    # If the LLM did not request another tool, we assume that it has
    # enough information to answer the user's question.
    #
    # The agent loop therefore ends here.
    #
    else:

        print(llm_response.content)

        break