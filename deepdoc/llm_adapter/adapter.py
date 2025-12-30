"""
LLM Adapter - Thin wrapper for LLM services

This module provides a unified interface for LLM services that works in both
FenixAOS environments and standalone configurations.
"""

import logging
from enum import Enum
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

# Try to import from FenixAOS first, fallback to local implementations
try:
    from fenixaos.core.model import LLMType as FenixLLMType
    from fenixaos.core.model.chat.basic.adapter import create_base_llm
    from fenixaos.core.model.model import ChatModelConfig, ImageModelConfig
    FENIXAOS_AVAILABLE = True
    logger.info("Using FenixAOS LLM services")
except ImportError:
    FENIXAOS_AVAILABLE = False
    logger.info("FenixAOS not available, using local LLM implementations")

# Local LLM implementations
try:
    from ..depend.simple_cv_model import create_vision_model
    LOCAL_VISION_AVAILABLE = True
except ImportError:
    LOCAL_VISION_AVAILABLE = False


class LLMType(str, Enum):
    """Unified LLM Type enumeration"""
    CHAT = "chat"
    EMBEDDING = "embedding"
    IMAGE2TEXT = "image2text"
    SPEECH2TEXT = "speech2text"
    RERANK = "rerank"

    @classmethod
    def from_fenix(cls, fenix_type: Any) -> 'LLMType':
        """Convert FenixAOS LLMType to unified type"""
        if FENIXAOS_AVAILABLE:
            # Map FenixAOS types to unified types
            type_mapping = {
                'chat': cls.CHAT,
                'embedding': cls.EMBEDDING,
                'image2text': cls.IMAGE2TEXT,
                'speech2text': cls.SPEECH2TEXT,
                'rerank': cls.RERANK,
            }
            return type_mapping.get(str(fenix_type).lower(), cls.CHAT)
        return cls.CHAT


class LLMServiceInterface:
    """Unified interface for LLM services"""

    def describe_with_prompt(self, image: Union[bytes, Any], prompt: Optional[str] = None) -> str:
        """Describe an image with optional prompt"""
        raise NotImplementedError

    def encode(self, texts: list) -> tuple:
        """Encode texts to embeddings"""
        raise NotImplementedError

    def chat(self, messages: list, **kwargs) -> str:
        """Chat completion"""
        raise NotImplementedError


class FenixAOSLLMService(LLMServiceInterface):
    """LLM Service adapter for FenixAOS environment"""

    def __init__(self, tenant_id: str, llm_type: LLMType, llm_name: Optional[str] = None, **kwargs):
        self.tenant_id = tenant_id
        self.llm_type = llm_type
        self.llm_name = llm_name

        # Try to create LLMBundle-like service from FenixAOS
        try:
            # This would need to be implemented based on FenixAOS's actual API
            self._service = self._create_fenix_service()
        except Exception as e:
            logger.warning(f"Failed to create FenixAOS LLM service: {e}")
            raise

    def _create_fenix_service(self):
        """Create service using FenixAOS APIs"""
        # Placeholder - needs to be implemented based on actual FenixAOS APIs
        raise NotImplementedError("FenixAOS service creation not implemented yet")

    def describe_with_prompt(self, image: Union[bytes, Any], prompt: Optional[str] = None) -> str:
        return self._service.describe_with_prompt(image, prompt)

    def encode(self, texts: list) -> tuple:
        return self._service.encode(texts)

    def chat(self, messages: list, **kwargs) -> str:
        return self._service.chat(messages, **kwargs)


class LocalLLMService(LLMServiceInterface):
    """LLM Service using local implementations"""

    def __init__(self, config: Optional[dict] = None, **kwargs):
        self.config = config or {}
        self._vision_model = None

        if LOCAL_VISION_AVAILABLE:
            try:
                self._vision_model = create_vision_model()
                logger.info("Local vision model initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize local vision model: {e}")

    def describe_with_prompt(self, image: Union[bytes, Any], prompt: Optional[str] = None) -> str:
        if self._vision_model:
            return self._vision_model.describe_with_prompt(image, prompt)
        else:
            logger.warning("No vision model available")
            return "Vision model not available"

    def encode(self, texts: list) -> tuple:
        # Placeholder for embedding functionality
        logger.warning("Local embedding not implemented")
        return [], 0

    def chat(self, messages: list, **kwargs) -> str:
        # Placeholder for chat functionality
        logger.warning("Local chat not implemented")
        return "Local chat not available"


class LLMAdapter:
    """Main adapter class that provides unified LLM service access"""

    def __init__(self, tenant_id: Optional[str] = None, llm_type: LLMType = LLMType.IMAGE2TEXT,
                 llm_name: Optional[str] = None, **kwargs):
        self.tenant_id = tenant_id
        self.llm_type = llm_type
        self.llm_name = llm_name
        self.kwargs = kwargs

        # Try FenixAOS first, then fallback to local
        self._service = self._create_service()

    def _create_service(self) -> LLMServiceInterface:
        """Create appropriate LLM service based on environment"""
        if FENIXAOS_AVAILABLE and self.tenant_id:
            try:
                return FenixAOSLLMService(self.tenant_id, self.llm_type, self.llm_name, **self.kwargs)
            except Exception as e:
                logger.warning(f"FenixAOS LLM service creation failed: {e}, falling back to local")

        # Fallback to local implementation
        return LocalLLMService(**self.kwargs)

    def describe_with_prompt(self, image: Union[bytes, Any], prompt: Optional[str] = None) -> str:
        """Describe an image with optional prompt"""
        return self._service.describe_with_prompt(image, prompt)

    def encode(self, texts: list) -> tuple:
        """Encode texts to embeddings"""
        return self._service.encode(texts)

    def chat(self, messages: list, **kwargs) -> str:
        """Chat completion"""
        return self._service.chat(messages, **kwargs)

    # Compatibility methods for LLMBundle-like interface
    def bind_tools(self, toolcall_session, tools):
        """Bind tools (placeholder for compatibility)"""
        logger.debug("Tool binding not implemented in adapter")

    @property
    def is_tools(self) -> bool:
        """Check if tools are supported"""
        return False

    @property
    def max_length(self) -> int:
        """Maximum context length"""
        return 4096  # Default value


def create_llm_service(tenant_id: Optional[str] = None, llm_type: LLMType = LLMType.IMAGE2TEXT,
                       llm_name: Optional[str] = None, **kwargs) -> LLMAdapter:
    """
    Factory function to create LLM service

    Args:
        tenant_id: Tenant ID (required for FenixAOS)
        llm_type: Type of LLM service
        llm_name: Specific model name
        **kwargs: Additional configuration

    Returns:
        LLMAdapter instance
    """
    return LLMAdapter(tenant_id, llm_type, llm_name, **kwargs)
