import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import onnxruntime as ort

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    Range,
)


load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "grounded_chunks"




model_path = hf_hub_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    filename="onnx/model.onnx"
)

tokenizer_path = hf_hub_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    filename="tokenizer.json"
)

tokenizer = Tokenizer.from_file(tokenizer_path)

tokenizer.enable_padding()
tokenizer.enable_truncation(max_length=256)

session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 1
session_options.inter_op_num_threads = 1
session_options.enable_cpu_mem_arena = False

session = ort.InferenceSession(
    model_path,
    sess_options=session_options,
    providers=["CPUExecutionProvider"]
)

print("Embeddings loaded successfully.")



client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

print("Qdrant connected successfully.")




def embed_query(text: str):

    encoding = tokenizer.encode(text)

    input_ids = np.array(
        [encoding.ids],
        dtype=np.int64
    )

    attention_mask = np.array(
        [encoding.attention_mask],
        dtype=np.int64
    )

    token_type_ids = np.array(
        [encoding.type_ids],
        dtype=np.int64
    )

    outputs = session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        }
    )

    token_embeddings = outputs[0]

    mask = attention_mask[..., None]

    summed = np.sum(
        token_embeddings * mask,
        axis=1
    )

    counts = np.sum(
        mask,
        axis=1
    )

    sentence_embedding = summed / np.clip(
        counts,
        a_min=1e-9,
        a_max=None
    )

    norm = np.linalg.norm(
        sentence_embedding,
        axis=1,
        keepdims=True
    )

    sentence_embedding = (
        sentence_embedding
        / np.clip(
            norm,
            a_min=1e-12,
            a_max=None
        )
    )

    return sentence_embedding[0].tolist()



def get_best_chunk(query: str):

    query_vector = embed_query(query)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=1,
        with_payload=True
    )

    if not results.points:
        return None

    return results.points[0]




def get_neighboring_chunks(best_chunk):

    if best_chunk is None:
        return []

    conversation_id = best_chunk.payload[
        "conversation_id"
    ]

    chunk_index = best_chunk.payload[
        "chunk_index"
    ]

    neighbors = client.scroll(
        collection_name=COLLECTION_NAME,

        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="conversation_id",
                    match=MatchValue(
                        value=conversation_id
                    )
                ),

                FieldCondition(
                    key="chunk_index",
                    range=Range(
                        gte=chunk_index - 1,
                        lte=chunk_index + 1
                    )
                )
            ]
        ),

        limit=3,
        with_payload=True
    )[0]

    neighbors = sorted(
        neighbors,
        key=lambda point:
            point.payload["chunk_index"]
    )

    return neighbors




def build_context(neighbors):

    texts = [
        point.payload["text"]
        for point in neighbors
    ]

    return "\n\n".join(texts)




def retrieve_context(query: str):

    best_chunk = get_best_chunk(query)

    if best_chunk is None:
        return ""

    neighbors = get_neighboring_chunks(
        best_chunk
    )

    context = build_context(
        neighbors
    )

    return context