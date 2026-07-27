import os
import sys

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.voyage.embedding.transformation_contextual import (
    VoyageContextualEmbeddingConfig,
)


class TestVoyageContextualDetection:
    def test_voyage_context_4_detected(self):
        config = VoyageContextualEmbeddingConfig()
        assert config.is_contextualized_embeddings("voyage-context-4") is True
        assert config.is_contextualized_embeddings("voyage/voyage-context-4") is True

    def test_non_contextual_models_not_detected(self):
        config = VoyageContextualEmbeddingConfig()
        assert config.is_contextualized_embeddings("voyage-4") is False
        assert config.is_contextualized_embeddings("voyage-4-nano") is False
        assert config.is_contextualized_embeddings("voyage-3.5") is False


class TestVoyageContextualInputNormalization:
    """
    The contextualized embeddings API expects ``inputs`` to be
    ``list[list[str]]`` (each document is a ``list[str]`` of chunks).
    """

    def test_flat_list_of_str_is_wrapped(self):
        config = VoyageContextualEmbeddingConfig()
        transformed = config.transform_embedding_request(
            "voyage-context-4", ["hello", "world"], {}, {}
        )
        assert transformed["inputs"] == [["hello", "world"]]
        assert all(isinstance(doc, list) for doc in transformed["inputs"])

    def test_single_string_is_wrapped(self):
        config = VoyageContextualEmbeddingConfig()
        transformed = config.transform_embedding_request(
            "voyage-context-4", "hello", {}, {}
        )
        assert transformed["inputs"] == [["hello"]]

    def test_nested_list_passed_through(self):
        config = VoyageContextualEmbeddingConfig()
        nested = [["a", "b"], ["c"]]
        transformed = config.transform_embedding_request(
            "voyage-context-4", nested, {}, {}
        )
        assert transformed["inputs"] == nested

    def test_model_and_optional_params_preserved(self):
        config = VoyageContextualEmbeddingConfig()
        transformed = config.transform_embedding_request(
            "voyage-context-4", ["x"], {"output_dimension": 512}, {}
        )
        assert transformed["model"] == "voyage-context-4"
        assert transformed["output_dimension"] == 512
        assert transformed["inputs"] == [["x"]]
