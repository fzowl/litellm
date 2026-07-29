"""
This module is used to transform the request and response for the Voyage contextualized embeddings API. 
This would be used for all the contextualized embeddings models in Voyage. 
"""
from typing import List, Optional, Union

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
        headers: Union[dict, httpx.Headers] = {},
    ):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(
            method="POST", url="https://api.voyageai.com/v1/contextualizedembeddings"
        )
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
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
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
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
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
        input: Union[AllEmbeddingInputValues, List[List[str]]],
        optional_params: dict,
        headers: dict,
    ) -> dict:
        inputs = self._transform_input(input)
        request: dict = {
            "inputs": inputs,
            "model": model,
            **optional_params,
        }
        # A flat ``List[str]`` *document* input is only accepted by the API with
        # auto-chunking enabled; queries take a flat list directly and nested
        # (pre-grouped) inputs need no extra params. Inject the document
        # defaults only for the flat-document case, never overriding a value the
        # caller already set (``setdefault``).
        if self._needs_auto_chunking(inputs, request.get("input_type")):
            request.setdefault("input_type", "document")
            request.setdefault("enable_auto_chunking", True)
        return request

    @staticmethod
    def _transform_input(
        input: Union[AllEmbeddingInputValues, List[List[str]]],
    ) -> Union[AllEmbeddingInputValues, List[List[str]]]:
        """
        Normalize ``input`` to the shape the Voyage contextual ``inputs`` field
        expects, *preferring the flat* ``List[str]`` form and only keeping the
        nested ``List[List[str]]`` when the caller already supplied pre-grouped
        chunks.

        The ``inputs`` field accepts ``Union[List[List[str]], List[str]]``::

            "Hello"                -> ["Hello"]              (flat)
            ["text1", "text2"]     -> ["text1", "text2"]     (flat, independent)
            [["c1", "c2"], ["d1"]] -> [["c1", "c2"], ["d1"]] (nested, kept as-is)

        Reference: https://docs.voyageai.com/docs/contextualized-chunk-embeddings
        """
        if isinstance(input, str):
            return [input]
        if isinstance(input, list):
            return input
        return [input]

    @staticmethod
    def _needs_auto_chunking(
        inputs: Union[AllEmbeddingInputValues, List[List[str]]],
        input_type: Optional[str],
    ) -> bool:
        """
        Whether ``inputs`` requires the document auto-chunking params.

        Only a flat ``List[str]`` embedded as documents needs them. Queries
        (``input_type="query"``) accept a flat list directly, and nested
        ``List[List[str]]`` (pre-grouped chunks) are valid as-is.
        """
        if input_type == "query":
            return False
        is_nested = (
            isinstance(inputs, list)
            and len(inputs) > 0
            and all(isinstance(item, list) for item in inputs)
        )
        return not is_nested

    def transform_embedding_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: EmbeddingResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str] = None,
        request_data: dict = {},
        optional_params: dict = {},
        litellm_params: dict = {},
    ) -> EmbeddingResponse:
        try:
            raw_response_json = raw_response.json()
        except Exception:
            raise VoyageError(
                message=raw_response.text, status_code=raw_response.status_code
            )

        # model_response.usage
        model_response.model = raw_response_json.get("model")
        model_response.data = raw_response_json.get("data")
        model_response.object = raw_response_json.get("object")

        usage = Usage(
            prompt_tokens=raw_response_json.get("usage", {}).get("total_tokens", 0),
            total_tokens=raw_response_json.get("usage", {}).get("total_tokens", 0),
        )
        model_response.usage = usage
        return model_response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return VoyageError(
            message=error_message, status_code=status_code, headers=headers
        )

    @staticmethod
    def is_contextualized_embeddings(model: str) -> bool:
        return "context" in model.lower()
