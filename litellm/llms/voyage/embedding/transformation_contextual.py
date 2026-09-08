"""
This module is used to transform the request and response for the Voyage contextualized embeddings API.
This would be used for all the contextualized embeddings models in Voyage.
"""

from typing import Final

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.utils import EmbeddingResponse, Usage


class VoyageError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: dict | httpx.Headers = {},
    ):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(method="POST", url="https://api.voyageai.com/v1/contextualizedembeddings")
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            status_code=status_code,
            message=message,
            headers=headers,
        )


class VoyageContextualEmbeddingConfig(BaseEmbeddingConfig):
    """
    Reference: https://docs.voyageai.com/reference/embeddings-api
    """

    def __init__(self) -> None:
        pass

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        if api_base:
            if not api_base.endswith("/contextualizedembeddings"):
                api_base = f"{api_base}/contextualizedembeddings"
            return api_base
        return "https://api.voyageai.com/v1/contextualizedembeddings"

    def get_supported_openai_params(self, model: str) -> list:
        return ["encoding_format", "dimensions"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI params to Voyage params

        Reference: https://docs.voyageai.com/reference/contextualized-embeddings-api
        """
        if "encoding_format" in non_default_params:
            optional_params["encoding_format"] = non_default_params["encoding_format"]
        if "dimensions" in non_default_params:
            optional_params["output_dimension"] = non_default_params["dimensions"]
        return optional_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        if api_key is None:
            api_key = (
                get_secret_str("VOYAGE_API_KEY")
                or get_secret_str("VOYAGE_AI_API_KEY")
                or get_secret_str("VOYAGE_AI_TOKEN")
            )
        return {
            "Authorization": f"Bearer {api_key}",
        }

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues | list[list[str]],
        optional_params: dict,
        headers: dict,
    ) -> dict:
        inputs, contextual_params = self._prepare_contextual_inputs(input, optional_params)
        return {
            "inputs": inputs,
            "model": model,
            **optional_params,
            **contextual_params,
        }

    @staticmethod
    def _prepare_contextual_inputs(
        input: AllEmbeddingInputValues | list[list[str]],
        optional_params: dict,
    ) -> tuple[AllEmbeddingInputValues | list[list[str]], dict[str, str | bool]]:
        """
        Shape ``inputs`` and the auto-chunking params to match Voyage's
        contextualized embeddings contract.

        - ``list[list[str]]`` (pre-chunked documents) is always valid and passes through.
        - A flat ``list[str]`` or bare ``str`` is only valid as documents when
          ``enable_auto_chunking=True`` with ``input_type="document"``, or as
          queries with ``input_type="query"``. So a non-query flat input is sent
          with those two params defaulted (caller-set values win).

        Reference: https://docs.voyageai.com/docs/contextualized-chunk-embeddings
        """
        if isinstance(input, list) and len(input) > 0 and isinstance(input[0], list):
            return input, {}
        flat: Final = [input] if isinstance(input, str) else input
        if optional_params.get("input_type") == "query":
            return flat, {}
        contextual_params: Final = {
            **({"input_type": "document"} if "input_type" not in optional_params else {}),
            **({"enable_auto_chunking": True} if "enable_auto_chunking" not in optional_params else {}),
        }
        return flat, contextual_params

    def transform_embedding_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: EmbeddingResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: str | None = None,
        request_data: dict = {},
        optional_params: dict = {},
        litellm_params: dict = {},
    ) -> EmbeddingResponse:
        try:
            raw_response_json: Final = raw_response.json()
        except Exception:
            raise VoyageError(message=raw_response.text, status_code=raw_response.status_code)

        # model_response.usage
        model_response.model = raw_response_json.get("model")
        model_response.data = raw_response_json.get("data")
        model_response.object = raw_response_json.get("object")

        usage: Final = Usage(
            prompt_tokens=raw_response_json.get("usage", {}).get("total_tokens", 0),
            total_tokens=raw_response_json.get("usage", {}).get("total_tokens", 0),
        )
        model_response.usage = usage
        return model_response

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        return VoyageError(message=error_message, status_code=status_code, headers=headers)

    @staticmethod
    def is_contextualized_embeddings(model: str) -> bool:
        return "context" in model.lower()
