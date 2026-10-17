import base64
import io
import json
import os
import uuid
from abc import ABC
from io import BytesIO
from urllib.parse import urljoin

import requests
from openai import OpenAI
from openai.lib.azure import AzureOpenAI
from PIL import Image
from zai import ZhipuAiClient

# 可选导入 ollama
try:
    from ollama import Client
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    Client = None

# 修复导入路径问题
try:
    from .prompts import vision_llm_describe_prompt
except ImportError:
    try:
        from ..depend.prompts import vision_llm_describe_prompt
    except ImportError:
        # 如果都失败，提供默认提示词
        vision_llm_describe_prompt = """请详细描述这张图片的内容，包括：
1. 图片中的主要对象和场景
2. 任何可见的文字内容
3. 图片的布局和结构
4. 如果有表格，请描述表格的结构和内容
5. 如果有图表，请描述图表的数据和趋势

请用中文回答，描述要准确、详细。"""


class Base(ABC):
    def __init__(self, key, model_name, lang="Chinese"):
        self.key = key
        self.model_name = model_name
        self.lang = lang

    def image2base64(self, binary):
        """将图片转换为base64编码"""
        if isinstance(binary, BytesIO):
            binary.seek(0)
            img_data = binary.read()
        elif isinstance(binary, bytes):
            img_data = binary
        else:
            raise ValueError("binary must be BytesIO or bytes")
        
        return base64.b64encode(img_data).decode('utf-8')

    def prompt(self, image):
        """生成提示词"""
        return self.vision_llm_prompt(image)

    def vision_llm_prompt(self, image):
        """生成视觉LLM提示词"""
        if self.lang.lower() == "chinese":
            prompt_text = vision_llm_describe_prompt()
        else:
            prompt_text = "Please describe this image in detail."
        
        return prompt_text


class GptV4(Base):
    _FACTORY_NAME = "OpenAI"

    def __init__(self, key, model_name="gpt-4-vision-preview", lang="Chinese", **kwargs):
        super().__init__(key, model_name, lang)
        self.client = OpenAI(api_key=key)

    def describe_with_prompt(self, image, prompt=None):
        """使用OpenAI GPT-4 Vision模型描述图片"""
        try:
            base64_image = self.image2base64(image)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt or self.prompt(image)
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1000
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            return "**ERROR**: " + str(e)


class QWenCV(Base):
    _FACTORY_NAME = "Tongyi-Qianwen"

    def __init__(self, key, model_name="qwen-vl-max", lang="Chinese", **kwargs):
        super().__init__(key, model_name, lang)
        # 使用兼容模式的OpenAI API
        self.client = OpenAI(
            api_key=key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def describe_with_prompt(self, image, prompt=None):
        """使用通义千问视觉模型描述图片"""
        try:
            base64_image = self.image2base64(image)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt or self.prompt(image)
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1000
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            return "**ERROR**: " + str(e)


class Zhipu4V(Base):
    _FACTORY_NAME = "ZhipuAI"

    def __init__(self, key, model_name="glm-4v", lang="Chinese", **kwargs):
        super().__init__(key, model_name, lang)
        # Preserve the legacy override and China default instead of allowing
        # the maintained SDK's ZAI_BASE_URL to redirect this provider adapter.
        self.client = ZhipuAiClient(
            api_key=key,
            base_url=kwargs.get("base_url") or os.getenv("ZHIPUAI_BASE_URL") or "https://open.bigmodel.cn/api/paas/v4/",
        )

    def describe_with_prompt(self, image, prompt=None):
        """使用智谱AI视觉模型描述图片"""
        try:
            base64_image = self.image2base64(image)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt or self.prompt(image)
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            return "**ERROR**: " + str(e)


class OllamaCV(Base):
    _FACTORY_NAME = "Ollama"

    def __init__(self, key, model_name, lang="Chinese", **kwargs):
        if not OLLAMA_AVAILABLE:
            raise ImportError("Ollama客户端未安装，请运行: pip install ollama")
        self.client = Client(host=kwargs.get("base_url", "http://localhost:11434"))
        self.model_name = model_name
        self.lang = lang

    def describe(self, image):
        prompt = self.prompt("")
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt[0]["content"][1]["text"],
                images=[image],
            )
            ans = response["response"].strip()
            return ans, 128
        except Exception as e:
            return "**ERROR**: " + str(e), 0

    def describe_with_prompt(self, image, prompt=None):
        vision_prompt = self.vision_llm_prompt("", prompt) if prompt else self.vision_llm_prompt("")
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=vision_prompt[0]["content"][1]["text"],
                images=[image],
            )
            ans = response["response"].strip()
            return ans, 128
        except Exception as e:
            return "**ERROR**: " + str(e), 0

    def chat(self, system, history, gen_conf, image=""):
        if system:
            history[-1]["content"] = system + history[-1]["content"] + "user query: " + history[-1]["content"]

        try:
            for his in history:
                if his["role"] == "user":
                    his["images"] = [image]
            options = {}
            if "temperature" in gen_conf:
                options["temperature"] = gen_conf["temperature"]
            if "top_p" in gen_conf:
                options["top_k"] = gen_conf["top_p"]
            if "presence_penalty" in gen_conf:
                options["presence_penalty"] = gen_conf["presence_penalty"]
            if "frequency_penalty" in gen_conf:
                options["frequency_penalty"] = gen_conf["frequency_penalty"]
            response = self.client.chat(
                model=self.model_name,
                messages=history,
                options=options,
                keep_alive=-1,
            )

            ans = response["message"]["content"].strip()
            return ans, response["eval_count"] + response.get("prompt_eval_count", 0)
        except Exception as e:
            return "**ERROR**: " + str(e), 0

    def chat_streamly(self, system, history, gen_conf, image=""):
        if system:
            history[-1]["content"] = system + history[-1]["content"] + "user query: " + history[-1]["content"]

        for his in history:
            if his["role"] == "user":
                his["images"] = [image]
        options = {}
        if "temperature" in gen_conf:
            options["temperature"] = gen_conf["temperature"]
        if "top_p" in gen_conf:
            options["top_k"] = gen_conf["top_p"]
        if "presence_penalty" in gen_conf:
            options["presence_penalty"] = gen_conf["presence_penalty"]
        if "frequency_penalty" in gen_conf:
            options["frequency_penalty"] = gen_conf["frequency_penalty"]
        ans = ""
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=history,
                stream=True,
                options=options,
                keep_alive=-1,
            )
            for resp in response:
                if resp["done"]:
                    yield resp.get("prompt_eval_count", 0) + resp.get("eval_count", 0)
                ans += resp["message"]["content"]
                yield ans
        except Exception as e:
            yield ans + "\n**ERROR**: " + str(e)
        yield 0


class GeminiCV(Base):
    _FACTORY_NAME = "Google-Gemini"

    def __init__(self, key, model_name="gemini-pro-vision", lang="Chinese", **kwargs):
        super().__init__(key, model_name, lang)
        import google.generativeai as genai
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(model_name)

    def describe_with_prompt(self, image, prompt=None):
        """使用Google Gemini模型描述图片"""
        try:
            if isinstance(image, BytesIO):
                image.seek(0)
                img_data = image.read()
            elif isinstance(image, bytes):
                img_data = image
            else:
                raise ValueError("image must be BytesIO or bytes")
            
            pil_image = Image.open(BytesIO(img_data))
            
            response = self.model.generate_content([
                prompt or self.prompt(image),
                pil_image
            ])

            return response.text.strip()
        except Exception as e:
            return "**ERROR**: " + str(e)


class AnthropicCV(Base):
    _FACTORY_NAME = "Anthropic"

    def __init__(self, key, model_name="claude-3-sonnet-20240229", lang="Chinese", **kwargs):
        super().__init__(key, model_name, lang)
        import anthropic
        self.client = anthropic.Anthropic(api_key=key)

    def describe_with_prompt(self, image, prompt=None):
        """使用Anthropic Claude模型描述图片"""
        try:
            base64_image = self.image2base64(image)
            
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt or self.prompt(image)
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": base64_image
                                }
                            }
                        ]
                    }
                ]
            )

            return message.content[0].text.strip()
        except Exception as e:
            return "**ERROR**: " + str(e)


class VisionModelFactory:
    """视觉模型工厂类"""
    
    _PROVIDERS = {
        "openai": GptV4,
        "qwen": QWenCV,
        "zhipu": Zhipu4V,
        "ollama": OllamaCV,
        "gemini": GeminiCV,
        "anthropic": AnthropicCV,
    }

    @classmethod
    def create_model(cls, config):
        """根据配置创建视觉模型实例"""
        provider = config.get("provider", "openai").lower()
        model_name = config.get("model_name", "")
        api_key = config.get("api_key", "")
        lang = config.get("lang", "Chinese")
        base_url = config.get("base_url", "")
        
        if provider not in cls._PROVIDERS:
            raise ValueError(f"不支持的提供商: {provider}")
        
        model_class = cls._PROVIDERS[provider]
        
        # 根据提供商设置默认模型名称
        if not model_name:
            if provider == "openai":
                model_name = "gpt-4-vision-preview"
            elif provider == "qwen":
                model_name = "qwen-vl-max"
            elif provider == "zhipu":
                model_name = "glm-4v"
            elif provider == "gemini":
                model_name = "gemini-pro-vision"
            elif provider == "anthropic":
                model_name = "claude-3-sonnet-20240229"
        
        kwargs = {"lang": lang}
        if base_url:
            kwargs["base_url"] = base_url
        
        return model_class(api_key, model_name, **kwargs)

    @classmethod
    def create_from_env(cls):
        """从环境变量创建模型"""
        config = {
            "provider": os.getenv("DEEPDOC_VISION_PROVIDER", "openai"),
            "model_name": os.getenv("DEEPDOC_VISION_MODEL", ""),
            "api_key": os.getenv("DEEPDOC_VISION_API_KEY", ""),
            "lang": os.getenv("DEEPDOC_VISION_LANG", "Chinese"),
            "base_url": os.getenv("DEEPDOC_VISION_BASE_URL", ""),
        }
        return cls.create_model(config)

    @classmethod
    def create_from_config_file(cls, config_file="deepdoc_config.yaml"):
        """从配置文件创建模型"""
        try:
            import yaml
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            vision_config = config.get("vision_model", {})
            return cls.create_model(vision_config)
        except Exception as e:
            raise ValueError(f"Failed to load config from {config_file}: {e}")


def create_vision_model(config=None):
    """创建视觉模型 - 支持多种配置方式"""
    if config is None:
        # 方式1：从环境变量创建
        return VisionModelFactory.create_from_env()
    
    elif isinstance(config, str):
        # 方式2：字符串配置
        if config.lower() in VisionModelFactory._PROVIDERS:
            # 字符串是提供商名称
            return VisionModelFactory.create_model({"provider": config.lower()})
        elif config.endswith(('.yaml', '.yml', '.json')):
            # 字符串是配置文件路径
            return VisionModelFactory.create_from_config_file(config)
        else:
            # 尝试作为配置文件路径处理
            try:
                return VisionModelFactory.create_from_config_file(config)
            except:
                raise ValueError(f"无效的配置: {config}。支持的提供商: {list(VisionModelFactory._PROVIDERS.keys())}")
    
    elif isinstance(config, dict):
        # 方式3：字典配置
        return VisionModelFactory.create_model(config)
    
    else:
        raise ValueError(f"不支持的配置类型: {type(config)}")
