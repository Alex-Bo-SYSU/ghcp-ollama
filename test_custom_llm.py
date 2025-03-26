import json
import os
from custom_llm import CustomLLM
from litellm.utils import ModelResponse

os.environ['OPENAI_API_KEY'] = "" # litellm reads OPENAI_API_KEY from .env and sends the request

# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": "celsius"})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": "celsius"})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


def test_parallel_function_call():
    try:
        # Step 1: send the conversation and available functions to the model
        llm = CustomLLM()
        messages = [{"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris?"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
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
        
        print("\nFirst LLM Response:\n", response)
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        print("\nLength of tool calls", len(tool_calls))

        # Step 2: check if the model wanted to call a function
        if tool_calls:
            # Step 3: call the function
            available_functions = {
                "get_current_weather": get_current_weather,
            }
            messages.append({
                "role": "assistant",
                "content": response_message.content,
                "tool_calls": [{
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                } for tool_call in tool_calls]
            })

            # Step 4: send the info for each function call and function response to the model
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(
                    location=function_args.get("location"),
                    unit=function_args.get("unit"),
                )
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )

            # 使用相同的 CustomLLM 实例进行第二次调用
            second_response = llm.completion(
                model="gpt-4o",
                messages=messages,
                api_base="https://api.individual.githubcopilot.com/chat/completions",
                custom_prompt_dict={},
                model_response=ModelResponse(),
                print_verbose=print,
                encoding=None,
                api_key="YOUR_API_KEY",
                logging_obj=None,
                optional_params={},  # 第二次调用不需要 functions
            )
            print("\nSecond LLM response:\n", second_response)
            return second_response
    except Exception as e:
        print(f"Error occurred: {e}")

test_parallel_function_call()