import logging
import asyncio
import random
import json
import re
import time
from typing import List, Dict, Any, Optional
# aiohttp is typically used for high concurrency async HTTP requests.
# If not installed, users should `pip install aiohttp`.
try:
    import aiohttp
except ImportError:
    aiohttp = None

logger = logging.getLogger(__name__)

class LLMAPIManager:
    """
    大模型 API 高并发调度器
    负责：API Key 轮询池化、并发限流 (Semaphore)、指数退避重试 (Exponential Backoff)
    """
    def __init__(self, api_config: Dict[str, Any]):
        """
        api_config 结构示例:
        {
            "keys": ["sk-xxx1", "sk-xxx2"],
            "base_url": "https://api.openai.com/v1/chat/completions",
            "model": "gpt-4",
            "max_concurrency": 5,
            "max_retries": 3,
            "timeout_seconds": 60
        }
        """
        if not aiohttp:
            raise ImportError("aiohttp is required for LLMAPIManager. Please run: pip install aiohttp")
            
        self.api_keys = api_config.get("keys", [])
        if not self.api_keys:
            raise ValueError("No API keys provided in configuration.")
            
        self.base_url = api_config.get("base_url", "https://api.openai.com/v1/chat/completions")
        self.model = api_config.get("model", "gpt-4")
        
        self.max_concurrency = api_config.get("max_concurrency", 5)
        self.max_retries = api_config.get("max_retries", 3)
        self.timeout = api_config.get("timeout_seconds", 60)
        
        # 信号量控制全局并发数
        self.semaphore = asyncio.Semaphore(self.max_concurrency)
        
    def _get_random_key(self) -> str:
        """从密钥池中随机挑选一个，摊薄单 Key 压力"""
        return random.choice(self.api_keys)

    def _redact_sensitive_text(self, value: Any, max_length: int = 500) -> str:
        """Return a bounded diagnostic string without credentials or bearer tokens."""
        text = str(value).replace("\r", " ").replace("\n", " ")
        for api_key in self.api_keys:
            if api_key:
                text = text.replace(api_key, "[REDACTED]")
        text = re.sub(
            r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+",
            "Bearer [REDACTED]",
            text,
        )
        text = re.sub(
            r"\bsk-[A-Za-z0-9_-]+",
            "sk-[REDACTED]",
            text,
        )
        text = re.sub(
            r'(?i)(["\']?(?:api[_-]?key|token|secret|password)["\']?\s*[:=]\s*)'
            r'(["\']?)[^,\s}\]]+\2',
            r"\1[REDACTED]",
            text,
        )
        return text[:max_length]

    async def _call_single_prompt(self, session: aiohttp.ClientSession, prompt: str, semaphore: asyncio.Semaphore) -> Optional[str]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        
        for attempt in range(self.max_retries):
            # 获取凭证并锁定并发槽位
            api_key = self._get_random_key()
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            async with semaphore:
                try:
                    async with session.post(self.base_url, headers=headers, json=payload, timeout=self.timeout) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data["choices"][0]["message"]["content"]
                            
                        elif response.status == 429:
                            # 触发限流，进入指数退避
                            wait_time = (2 ** attempt) + random.uniform(0, 1)
                            logger.warning(
                                "Rate limited (429). Retrying in %.2fs",
                                wait_time,
                            )
                            await asyncio.sleep(wait_time)
                            
                        else:
                            error_text = await response.text()
                            logger.error(
                                "LLM API error %s: %s",
                                response.status,
                                self._redact_sensitive_text(error_text),
                            )
                            # 遇到非 429 错误也尝试退避，可能是网关错误
                            await asyncio.sleep(2 ** attempt)
                            
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout on attempt {attempt+1}/{self.max_retries}")
                    await asyncio.sleep(1)
                except Exception as e:
                    logger.error(
                        "Unexpected error calling LLM API (%s): %s",
                        type(e).__name__,
                        self._redact_sensitive_text(e),
                    )
                    await asyncio.sleep(1)
                    
        logger.error("Max retries exceeded for prompt.")
        return None

    async def _batch_generate_async(self, prompts: List[str]) -> List[str]:
        semaphore = asyncio.Semaphore(self.max_concurrency)
        async with aiohttp.ClientSession() as session:
            tasks = [self._call_single_prompt(session, p, semaphore) for p in prompts]
            results = await asyncio.gather(*tasks)
            return results

    def batch_generate(self, prompts: List[str]) -> List[str]:
        """
        同步调用的入口面，供外部普通 Python 代码直接调用
        """
        # 判断当前是否已在事件循环中
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
            
        if loop and loop.is_running():
            # 如果在 async 环境中调用，建议外部直接使用 await _batch_generate_async
            # 这里退化为阻塞调用供非异步框架兼容
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(lambda: asyncio.run(self._batch_generate_async(prompts))).result()
        else:
            return asyncio.run(self._batch_generate_async(prompts))
