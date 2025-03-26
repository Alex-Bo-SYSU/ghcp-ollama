from typing import Any, AsyncIterator, Iterator, Optional, Union, Callable
import json
import httpx
import logging
import time
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.utils import GenericStreamingChunk, Message, ChatCompletionMessageToolCall, Choices
from litellm.utils import ImageResponse, ModelResponse
from litellm.llms.base import BaseLLM

logging.basicConfig(level=logging.INFO)

class CustomLLMError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)

import requests

class CustomLLM(BaseLLM):
    def __init__(self) -> None:
        super().__init__()
        self.api_url = "https://api.individual.githubcopilot.com/chat/completions"
        self.default_model = "gpt-4o"
        self.headers = {
            'Content-Type': 'application/json',
            'Copilot-Integration-Id': 'vscode-chat',
            'Editor-Version': 'Neovim/0.10.3'
        }
        self.function_registry = {
            "get_weather": self.get_weather
        }
        self.last_token_refresh = 0
        self.token_refresh_interval = 300  # 5 minutes in seconds
        # Initialize token
        self.update_token()

    def update_token(self) -> None:
        """Update the Authorization token if needed (every 5 minutes)"""
        current_time = time.time()
        if current_time - self.last_token_refresh >= self.token_refresh_interval:
            token = get_copilot_token()
            if token:
                self.headers['Authorization'] = f'Bearer {token}'
                self.last_token_refresh = current_time
            else:
                raise CustomLLMError(401, "Failed to get valid Copilot token")

    def send_request(self, payload: dict) -> dict:
        try:
            # Update token only if needed
            self.update_token()
            
            formatted_payload = {
                "model": payload.get("model", self.default_model),
                "messages": payload.get("messages", []),
                "temperature": payload.get("temperature", 0.5),
                "max_tokens": payload.get("max_tokens", 8192),
                "functions": [
                    {
                        "name": func["function"]["name"],
                        "description": func["function"]["description"],
                        "parameters": func["function"]["parameters"]
                    }
                    for func in payload.get("functions", [])
                ]
            }
            print(f"Sending Url: {self.api_url}\nheaders: {self.headers}\ndata: {json.dumps(formatted_payload, indent=2)}")
            response = requests.post(self.api_url, headers=self.headers, json=formatted_payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise CustomLLMError(e.response.status_code, e.response.text)
        except Exception as e:
            raise CustomLLMError(500, str(e))

    def get_weather(self, location: str) -> dict:
        return {
            "location": location,
            "temperature": "22°C",
            "condition": "Partly cloudy"
        }

    def handle_function_calls(self, function_calls, messages):
        if not function_calls:
            return messages
        
        function_responses = []
        for function_call in function_calls:
            if isinstance(function_call, dict):
                function_name = function_call.get("function", {}).get("name")
                arguments = json.loads(function_call.get("function", {}).get("arguments", "{}"))
            else:
                function_name = function_call.get("name")
                arguments = json.loads(function_call.get("arguments", "{}"))
            
            if function_name in self.function_registry:
                result = self.function_registry[function_name](**arguments)
                function_responses.append({
                    "tool_call_id": function_call.get("id"),
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(result)
                })
        
        messages.extend(function_responses)
        return messages

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        **kwargs
    ) -> ModelResponse:
        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": optional_params.get("temperature", 0.5),
            "max_tokens": optional_params.get("maxTokens", 8192),
            "functions": optional_params.get("functions", [])
        }
        response_data = self.send_request(payload)
        raw_choices = response_data.get("choices", [])
        
        if raw_choices:
            raw_choice = raw_choices[0]
            raw_message = raw_choice.get("message", {})
            
            # 创建Message实例，设置基本属性
            response_message = Message(
                content=raw_message.get("content"),
                role="assistant"
            )
            
            # 处理function_call
            function_call = raw_message.get("function_call")
            if function_call:
                tool_call = ChatCompletionMessageToolCall(
                    function={
                        "name": function_call.get("name"),
                        "arguments": function_call.get("arguments", "{}")
                    },
                    type="function"
                )
                response_message.tool_calls = [tool_call]
                messages = self.handle_function_calls([tool_call], messages)
                
                # 创建Choices实例
                choice = Choices(
                    finish_reason=raw_choice.get("finish_reason", "function_call"),
                    index=raw_choice.get("index", 0),
                    message=response_message
                )
                
                model_response.choices = [choice]
                return model_response
            
            # 处理普通响应
            choice = Choices(
                finish_reason=raw_choice.get("finish_reason", "stop"),
                index=raw_choice.get("index", 0),
                message=response_message
            )
            
            model_response.choices = [choice]
        return model_response

    async def acompletion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        **kwargs
    ) -> ModelResponse:
        async with httpx.AsyncClient() as client:
            while True:
                payload = {
                    "model": model or self.default_model,
                    "messages": messages,
                    "temperature": optional_params.get("temperature", 0.5),
                    "max_tokens": optional_params.get("maxTokens", 8192),
                    "functions": optional_params.get("functions", [])
                }
                response = await client.post(self.api_url, headers=self.headers, json=payload)
                response_data = response.json()
                choices = response_data.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    tool_calls = message.get("tool_calls", []) or [message.get("function_call")] if message.get("function_call") else []
                    if tool_calls:
                        messages = self.handle_function_calls(tool_calls, messages)
                        continue  # 继续循环直到没有新的 function call
                model_response.choices = choices
                return model_response

    def streaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        **kwargs
    ) -> Iterator[GenericStreamingChunk]:
        raise CustomLLMError(status_code=500, message="Streaming not implemented yet!")

    async def astreaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        **kwargs
    ) -> AsyncIterator[GenericStreamingChunk]:
        raise CustomLLMError(status_code=500, message="Streaming not implemented yet!")


def custom_chat_llm_router(
    async_fn: bool, stream: Optional[bool], custom_llm: CustomLLM
):
    if async_fn:
        if stream:
            return custom_llm.astreaming
        return custom_llm.acompletion
    if stream:
        return custom_llm.streaming
    return custom_llm.completion


def test_weather_function_call():
    llm = CustomLLM()
    
    messages = [{"role": "user", "content": "What's the weather like in New York?"}]
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. New York, NY",
                        }
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    
    response = llm.completion(
        model="gpt-4o",
        messages=messages,
        api_base="https://api.individual.githubcopilot.com/chat/completions",
        custom_prompt_dict={},
        model_response=ModelResponse(),
        print_verbose=print,
        encoding=None,
        api_key="YOUR_API_KEY",
        logging_obj=None,
        optional_params={"functions": tools},
    )
    
    print("Response:", response.json)


import requests

def get_copilot_token():
    try:
        response = requests.get('http://localhost:11434/api/token')
        data = response.json()
        
        if data.get('success'):
            return data['token']
        else:
            raise Exception(data.get('error', 'Unknown error'))
            
    except Exception as e:
        print(f"Error getting Copilot token: {str(e)}")
        return None

test_weather_function_call()