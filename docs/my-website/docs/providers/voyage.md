# Voyage AI
https://docs.voyageai.com/embeddings/

## API Key
```python
# env variable
os.environ['VOYAGE_API_KEY']
```

## Sample Usage - Embedding
```python
from litellm import embedding
import os

os.environ['VOYAGE_API_KEY'] = ""
response = embedding(
    model="voyage/voyage-3.5",
    input=["good morning from litellm"],
)
print(response)
```

## Supported Parameters

VoyageAI embeddings support the following optional parameters:

- `input_type`: Specifies the type of input for retrieval optimization
  - `"query"`: Use for search queries
  - `"document"`: Use for documents being indexed
- `dimensions`: Output embedding dimensions (256, 512, 1024, or 2048)
- `encoding_format`: Output format (`"float"`, `"int8"`, `"uint8"`, `"binary"`, `"ubinary"`)
- `truncation`: Whether to truncate inputs exceeding max tokens (default: `True`)

### Example with Parameters

```python
from litellm import embedding
import os

os.environ['VOYAGE_API_KEY'] = "your-api-key"

# Embedding with custom dimensions and input type
response = embedding(
    model="voyage/voyage-3.5",
    input=["Your text here"],
    dimensions=512,
    input_type="document"
)
print(f"Embedding dimensions: {len(response.data[0]['embedding'])}")
```

## Supported Models
All models listed here https://docs.voyageai.com/embeddings/#models-and-specifics are supported

| Model Name              | Function Call                                              |
|-------------------------|------------------------------------------------------------|
| voyage-4-large          | `embedding(model="voyage/voyage-4-large", input)`          | 
| voyage-4                | `embedding(model="voyage/voyage-4", input)`                | 
| voyage-4-lite           | `embedding(model="voyage/voyage-4-lite", input)`           | 
| voyage-4-nano           | `embedding(model="voyage/voyage-4-nano", input)`           | 
| voyage-3.5              | `embedding(model="voyage/voyage-3.5", input)`              | 
| voyage-3.5-lite         | `embedding(model="voyage/voyage-3.5-lite", input)`         | 
| voyage-3-large          | `embedding(model="voyage/voyage-3-large", input)`          | 
| voyage-3                | `embedding(model="voyage/voyage-3", input)`                | 
| voyage-3-lite           | `embedding(model="voyage/voyage-3-lite", input)`           | 
| voyage-code-3           | `embedding(model="voyage/voyage-code-3", input)`           | 
| voyage-finance-2        | `embedding(model="voyage/voyage-finance-2", input)`        | 
| voyage-law-2            | `embedding(model="voyage/voyage-law-2", input)`            | 
| voyage-code-2           | `embedding(model="voyage/voyage-code-2", input)`           | 
| voyage-multilingual-2   | `embedding(model="voyage/voyage-multilingual-2	", input)`  | 
| voyage-large-2-instruct | `embedding(model="voyage/voyage-large-2-instruct", input)` | 
| voyage-large-2          | `embedding(model="voyage/voyage-large-2", input)`          |
| voyage-2                | `embedding(model="voyage/voyage-2", input)`                | 
| voyage-lite-02-instruct | `embedding(model="voyage/voyage-lite-02-instruct", input)` | 
| voyage-01               | `embedding(model="voyage/voyage-01", input)`               | 
| voyage-lite-01          | `embedding(model="voyage/voyage-lite-01", input)`          |
| voyage-lite-01-instruct | `embedding(model="voyage/voyage-lite-01-instruct", input)` |

:::note voyage-4-nano
`voyage-4-nano` is also published as an **open-weight** model on [Hugging Face](https://huggingface.co/voyageai). It is registered in the LiteLLM cost map at **$0/M tokens** (free); update the price if your deployment is billed differently.
:::

## Contextual Embeddings (voyage-context-4, voyage-context-3)

VoyageAI's contextual models (`voyage-context-4`, `voyage-context-3`) provide contextualized chunk embeddings, where each chunk is embedded with awareness of its surrounding document context. This significantly improves retrieval quality compared to standard context-agnostic embeddings.

### Key Benefits
- Chunks understand their position and role within the full document
- Improved retrieval accuracy for long documents (outperforms competitors by 7-23%)
- Better handling of ambiguous references and cross-chunk dependencies
- Seamless drop-in replacement for standard embeddings in RAG pipelines

### Usage

The Voyage contextual `inputs` field accepts **both** a flat `List[str]` and a nested `List[List[str]]` (`Union[List[List[str]], List[str]]`). LiteLLM **prefers the flat `List[str]`** form whenever the input allows it and only falls back to the nested `List[List[str]]` when you supply pre-grouped chunks yourself:

| You pass | LiteLLM sends (`inputs`) | Extra params |
|----------|--------------------------|--------------|
| `"Hello"` (a string) | `["Hello"]` (flat) | `input_type="document"`, `enable_auto_chunking=True` |
| `["text1", "text2"]` (flat list of independent texts) | `["text1", "text2"]` (flat, kept) | `input_type="document"`, `enable_auto_chunking=True` |
| `["query"]` with `input_type="query"` | `["query"]` (flat) | none |
| `[["c1", "c2"], ["d1"]]` (already nested — one list per document) | `[["c1", "c2"], ["d1"]]` (unchanged) | none |

Why the extra params: a flat `List[str]` document input is only valid when auto-chunking is enabled, so LiteLLM sets `input_type="document"` + `enable_auto_chunking=True` for the flat document case (Voyage splits each string into chunks server-side). Queries (`input_type="query"`) accept a flat list directly, so nothing extra is added. When you pass a nested `List[List[str]]` (pre-grouped chunks that should share context), it is forwarded untouched and **no** auto-chunking params are injected. Any `input_type` / `enable_auto_chunking` you set explicitly is respected.

```python
from litellm import embedding
import os

os.environ['VOYAGE_API_KEY'] = "your-api-key"

# Flat list of documents (sent as List[str], auto-chunked server-side)
response = embedding(
    model="voyage/voyage-context-4",
    input=[
        "Chapter 1: Introduction to AI. This chapter covers the basics.",
        "Chapter 2: Machine learning and deep learning fundamentals."
    ]
)
print(f"Number of results: {len(response.data)}")

# Pre-grouped chunks — one inner list per document, embedded in shared context
# (sent as-is: List[List[str]])
response = embedding(
    model="voyage/voyage-context-4",
    input=[
        ["Paris is the capital of France.", "It is known for the Eiffel Tower."],
        ["Tokyo is the capital of Japan.", "It is a major economic hub."]
    ]
)
print(f"Processed {len(response.data)} documents")
```

### Specifications
- Models: `voyage-context-4` ($0.12/M tokens), `voyage-context-3` ($0.18/M tokens)
- Per-chunk context window: 32,000 tokens
- Output dimensions: 256, 512, 1024 (default), or 2048
- Max inputs: 1,000 per request
- Max total tokens: 120,000
- Max chunks: 16,000

### When to Use Contextual Embeddings

**Use `voyage-context-4` / `voyage-context-3` when:**
- Processing long documents split into chunks
- Document structure and flow are important
- References between sections matter
- You need to preserve document hierarchy

**Use standard models (voyage-3.5, voyage-3-large) when:**
- Embedding independent pieces of text
- Processing short queries
- Document context is not relevant
- You need faster/cheaper processing

## Model Selection Guide

| Model | Best For | Context Length | Price/M Tokens |
|-------|----------|----------------|----------------|
| voyage-4-large | Best overall quality | 32K | $0.12 |
| voyage-4 | General-purpose, multilingual | 32K | $0.06 |
| voyage-4-lite | Latency-sensitive applications | 32K | $0.02 |
| voyage-4-nano | Smallest, open-weight | 32K | $0.00 |
| voyage-3.5 | General-purpose, multilingual | 32K | $0.06 |
| voyage-3.5-lite | Latency-sensitive applications | 32K | $0.02 |
| voyage-3-large | Best overall quality | 32K | $0.18 |
| voyage-code-3 | Code retrieval and search | 32K | $0.18 |
| voyage-finance-2 | Financial documents | 32K | $0.12 |
| voyage-law-2 | Legal documents | 16K | $0.12 |
| voyage-context-4 | Contextual document embeddings | 32K | $0.12 |
| voyage-context-3 | Contextual document embeddings | 32K | $0.18 |

## Rerank

Voyage AI provides reranking models to improve search relevance by reordering documents based on their relevance to a query.

### Quick Start

```python
from litellm import rerank
import os

os.environ["VOYAGE_API_KEY"] = "your-api-key"

response = rerank(
    model="voyage/rerank-2.5",
    query="What is the capital of France?",
    documents=[
        "Paris is the capital of France.",
        "London is the capital of England.",
        "Berlin is the capital of Germany.",
    ],
    top_n=3,
)

print(response)
```

### Async Usage

```python
from litellm import arerank
import os
import asyncio

os.environ["VOYAGE_API_KEY"] = "your-api-key"

async def main():
    response = await arerank(
        model="voyage/rerank-2.5-lite",
        query="Best programming language for beginners?",
        documents=[
            "Python is great for beginners due to simple syntax.",
            "JavaScript runs in browsers and is versatile.",
            "Rust has a steep learning curve but is very safe.",
        ],
        top_n=2,
    )
    print(response)

asyncio.run(main())
```

### LiteLLM Proxy Usage

Add to your `config.yaml`:

```yaml
model_list:
  - model_name: rerank-2.5
    litellm_params:
      model: voyage/rerank-2.5
      api_key: os.environ/VOYAGE_API_KEY
  - model_name: rerank-2.5-lite
    litellm_params:
      model: voyage/rerank-2.5-lite
      api_key: os.environ/VOYAGE_API_KEY
```

Test with curl:

```bash
curl http://localhost:4000/rerank \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rerank-2.5",
    "query": "What is the capital of France?",
    "documents": [
        "Paris is the capital of France.",
        "London is the capital of England.",
        "Berlin is the capital of Germany."
    ],
    "top_n": 3
  }'
```

### Supported Rerank Models

| Model | Context Length | Description | Price/M Tokens |
|-------|----------------|-------------|----------------|
| rerank-2.5 | 32K | Best quality, multilingual, instruction-following | $0.05 |
| rerank-2.5-lite | 32K | Optimized for latency and cost | $0.02 |
| rerank-2 | 16K | Legacy model | $0.05 |
| rerank-2-lite | 8K | Legacy model, faster | $0.02 |

### Supported Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | Model name (e.g., `voyage/rerank-2.5`) |
| `query` | string | The search query |
| `documents` | list | List of documents to rerank |
| `top_n` | int | Number of top results to return |
| `return_documents` | bool | Whether to include document text in response |
