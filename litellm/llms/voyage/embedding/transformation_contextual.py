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
        inputs, extra_params = self._transform_input(input, optional_params)
        return {
            "inputs": inputs,
            "model": model,
            **optional_params,
            **extra_params,
        }

    @staticmethod
    def _transform_input(
        input: Union[AllEmbeddingInputValues, List[List[str]]],
        optional_params: dict,
    ) -> tuple:
        """
        Normalize ``input`` for the Voyage contextualized embeddings API
        (``inputs`` field), *preferring the flat* ``List[str]`` form and only
        falling back to nested ``List[List[str]]`` when the caller already
        supplied pre-grouped chunks.

        The contextual ``inputs`` field accepts both shapes
        (``Union[List[List[str]], List[str]]``):

        - flat ``List[str]`` of independent texts — the idiomatic batch form.
          Valid when embedding queries (``input_type="query"``), or documents
          with ``enable_auto_chunking=True`` + ``input_type="document"``. A flat
          document list with auto-chunking disabled is rejected by the API, so
          for the document/unspecified case we set those two params (unless the
          caller already provided them) to keep the flat call valid.
        - nested ``List[List[str]]`` — one inner list per document, used when the
          caller pre-groups chunks that should share context. Forwarded as-is;
          no auto-chunking params are injected.

        Mapping::

            "Hello"                -> ["Hello"]              (flat)
            ["text1", "text2"]     -> ["text1", "text2"]     (flat, independent)
            [["c1", "c2"], ["d1"]] -> [["c1", "c2"], ["d1"]] (nested, kept as-is)

        Returns a ``(inputs, extra_params)`` tuple; ``extra_params`` carries any
        params that must accompany the chosen ``inputs`` shape.

        Reference: https://docs.voyageai.com/docs/contextualized-chunk-embeddings
        """
        # Already nested (List[List[...]]) -> pre-grouped chunks, valid as-is.
        if (
            isinstance(input, list)
            and len(input) > 0
            and all(isinstance(item, list) for item in input)
        ):
            return input, {}

        # Otherwise, normalize to the preferred flat List[str].
        if isinstance(input, str):
            flat: list = [input]
        elif isinstance(input, list):
            flat = input
        else:
            flat = [input]

        # A flat query list is accepted directly; no extra params needed.
        if optional_params.get("input_type") == "query":
            return flat, {}

        # A flat document list is only valid with auto-chunking enabled, so add
        # the required params unless the caller already set them explicitly.
        extra_params: dict = {}
        if "input_type" not in optional_params:
            extra_params["input_type"] = "document"
        if "enable_auto_chunking" not in optional_params:
            extra_params["enable_auto_chunking"] = True
        return flat, extra_params

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
