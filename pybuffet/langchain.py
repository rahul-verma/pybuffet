import os
import openai
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import ToolMessage

class LangChainClient:
    
    def __init__(self, system_prompt="You are a helpful AI assistant.", model="gpt-4o-mini"):
        self.__model = model
        openai.api_key  = os.getenv('OPENAI_API_KEY')
        self.__system_prompt = system_prompt

    def run_prompt(self, prompt, **kwargs):
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.__system_prompt),
            ("human", prompt)
        ])
            
        client = ChatOpenAI(
            model=self.__model,
            temperature=0
        )
        
        chain = prompt_template | client
        
        return chain.invoke(kwargs)
    

class LangChainClientWithMemory(LangChainClient):
    
    def __init__(self, system_prompt="You are a helpful AI assistant.", model="gpt-4o-mini"):
        super().__init__(system_prompt, model)
        # A simple in-memory store for demonstration
        self.__store = {}
        
        base_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}")
            ]
        )
        
        llm = ChatOpenAI(
            model=model,
            temperature=0
        )
        
        chain = base_prompt | llm
        
        self.__chain_with_history = RunnableWithMessageHistory(
            chain,
            get_session_history=self.__get_session_history,
            input_messages_key="question",  # Key for the current user input
            history_messages_key="history"  # Key where history will be placed in the prompt
        )

    def __get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.__store:
            self.__store[session_id] = ChatMessageHistory()
        return self.__store[session_id]

    def run_prompt(self, prompt, **kwargs):
        formatted_prompt = ChatPromptTemplate.from_messages([("human", prompt)]).format(**kwargs)
        return self.__chain_with_history.invoke(
            {"question": formatted_prompt},
            config={"configurable": {"session_id": "user123"}}
        )


class LangChainClientWithTools(LangChainClient):
    
    def __init__(self, tools, system_prompt="You are a helpful AI assistant.", model="gpt-4o-mini"):
        super().__init__(system_prompt, model)
        # A simple in-memory store for demonstration
        self.__store = {}
        
        base_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}")
            ]
        )

        llm = ChatOpenAI(
            model=model,
            temperature=0
        )
        
        llm_with_tools = llm.bind_tools(tools, tool_choice="auto")
        
        chain = base_prompt | llm_with_tools
        
        self.__chain_with_history = RunnableWithMessageHistory(
            chain,
            get_session_history=self.__get_session_history,
            input_messages_key="question",  # Key for the current user input
            history_messages_key="history"  # Key where history will be placed in the prompt
        )

    def __get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.__store:
            self.__store[session_id] = ChatMessageHistory()
        return self.__store[session_id]
    
    def run_prompt(self, prompt: str, tools_dict, session_id: str="tester", **kwargs) -> str:

        formatted_prompt = prompt.format(**kwargs)
        
        # 1st model call: may include tool_calls
        ai_msg = self.__chain_with_history.invoke(
            {"question": formatted_prompt},
            config={"configurable": {"session_id": session_id}}
        )

        # Tool loop
        print("Starting the tool loop...")
        if hasattr(ai_msg, "tool_calls"):
            tool_messages = []
            for tc in ai_msg.tool_calls:
                tool = tools_dict[tc["name"]]
                tool_args = tc["args"]
                print("Invoking tool:", tc["name"], "with args:", tool_args)
                result = tool.invoke(tool_args)
                tool_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            
            # Persist: AI request + tool replies
            history = self.__get_session_history(session_id)
            for tm in tool_messages:
                history.add_message(tm)
                
            # Follow-up model call – do NOT add a new human turn unless you have one
            return self.__chain_with_history.invoke(
                {"question": "Now provide the answer."},  # empty human message; model will read tool outputs from history
                config={"configurable": {"session_id": session_id}}
            )
        
