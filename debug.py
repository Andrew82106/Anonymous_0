from llms.manager import LLMManager

llm_manager = LLMManager()

print(llm_manager.list_models())

# 测试文本对话能力

# test_model_name = "gpt-4-turbo"
# test_prompt = "Hello, how are you?"
# print(llm_manager.call_model(test_model_name, test_prompt))

# 测试结构化输出能力
from pydantic import BaseModel

class TestResponse(BaseModel):
    message: str
    status: str

test_model_name = "gpt-4-turbo"
test_prompt = "Hello, how are you? Please respond with a message and status."
print(llm_manager.call_structured(test_model_name, test_prompt, TestResponse))