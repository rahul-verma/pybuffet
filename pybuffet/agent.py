import os
import openai
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import ToolMessage
from langchain.agents import create_react_agent, AgentExecutor
from typing import Dict


REACT_PROMPT_SUFFIX = """

# TOOLS
{tools}

You may call only these tool names: {tool_names}

# STRICT FORMAT (no code fences)

    Question: <the user question>
    Thought: <your reasoning>
    Action: <ONE tool name from {tool_names}>
    Action Input: <a JSON string or plain string for the tool>
    Observation: <tool result>
    ... (repeat Thought/Action/Action Input/Observation as needed) ...
    Thought: I now know the final answer.
    Final Answer: <your final answer>

# RULES:

    - Do NOT put parentheses on the Action line. Example: write 'Action: search_srs', NOT 'Action: search_srs(...)'
    - Put the parameter ONLY on the next line after 'Action Input:'.
    - Never write 'Action:' unless you are calling a tool.
    - Do not wrap the Action/Action Input in markdown code blocks.

"""


class LangChainAgent:
    
    def __init__(self, tools, system_prompt="You are a helpful AI assistant.", model="gpt-4o-mini"):
        # A simple in-memory store for demonstration
        self.__store: Dict[str, ChatMessageHistory] = {}

        final_system_prompt = "# MISSION\n" + system_prompt + REACT_PROMPT_SUFFIX
        
        base_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", final_system_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
                ("assistant", "{agent_scratchpad}")
            ]
        )

        llm = ChatOpenAI(
            model=model,
            temperature=0
        )
        
        agent = create_react_agent(llm, tools, base_prompt)
        
        tool_names=", ".join([tool.name for tool in tools])
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            # If the model misformats, tell it how to fix the last message
            handle_parsing_errors=(
                f"""Invalid format. Respond ONLY with:
                Action: <one of {tool_names}>
                Action Input: <string or JSON>
                Do not add anything else.
                """
            ),
            max_iterations=20,
            return_intermediate_steps=True,  # <— needed for eval
        )
        
        self.__agent_with_history = RunnableWithMessageHistory(
            agent_executor,
            get_session_history=self.__get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        
        self.__formatted_system_prompt = final_system_prompt.format(
            tool_names=", ".join([tool.name for tool in tools]),
            tools="\n\n".join([f"{tool.name}: {tool.description}" for tool in tools])
        )
        
    @property
    def system_prompt(self):
        return self.__formatted_system_prompt

    def __get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.__store:
            self.__store[session_id] = ChatMessageHistory()
        return self.__store[session_id]
    
    def run_prompt(self, prompt: str, session_id: str="tester", **kwargs) -> str:

        formatted_prompt = prompt.format(**kwargs)

        result = self.__agent_with_history.invoke(
            {"question": formatted_prompt},
            config={"configurable": {"session_id": session_id}}
        )
        
        return result["output"]
        
