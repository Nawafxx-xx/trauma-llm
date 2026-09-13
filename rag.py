import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import urllib.request

import numpy as np
import onnxruntime as ort

from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

class MiniLMONNXEmbeddings(Embeddings):
    def __init__(self):
        model_path = hf_hub_download(
            repo_id="sentence-transformers/all-MiniLM-L6-v2",
            filename="onnx/model.onnx"
        )

        tokenizer_path = hf_hub_download(
            repo_id="sentence-transformers/all-MiniLM-L6-v2",
            filename="tokenizer.json"
        )

        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        self.tokenizer.enable_padding()
        self.tokenizer.enable_truncation(max_length=256)

        self.session = ort.InferenceSession(model_path)

    def _embed(self, texts):
        encodings = self.tokenizer.encode_batch(texts)

        input_ids = np.array(
            [encoding.ids for encoding in encodings],
            dtype=np.int64
        )

        attention_mask = np.array(
            [encoding.attention_mask for encoding in encodings],
            dtype=np.int64
        )

        token_type_ids = np.array(
            [encoding.type_ids for encoding in encodings],
            dtype=np.int64
        )

        outputs = self.session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids
            }
        )

        token_embeddings = outputs[0]

        mask = attention_mask[..., None]

        summed_embeddings = np.sum(
            token_embeddings * mask,
            axis=1
        )

        token_counts = np.sum(
            mask,
            axis=1
        )

        token_counts = np.clip(
            token_counts,
            a_min=1e-9,
            a_max=None
        )

        sentence_embeddings = (
            summed_embeddings / token_counts
        )

        norms = np.linalg.norm(
            sentence_embeddings,
            axis=1,
            keepdims=True
        )

        norms = np.clip(
            norms,
            a_min=1e-12,
            a_max=None
        )

        sentence_embeddings = (
            sentence_embeddings / norms
        )

        return sentence_embeddings.tolist()

    def embed_documents(self, texts):
        return self._embed(texts)

    def embed_query(self, text):
        return self._embed([text])[0]


embeddings = MiniLMONNXEmbeddings()
print("Embeddings loaded successfully.")


FAISS_FOLDER = "index_faiss"

FAISS_URL = "https://github.com/Nawafxx-xx/trauma-llm/releases/download/v1.0-index/index.faiss"
PKL_URL = "https://github.com/Nawafxx-xx/trauma-llm/releases/download/v1.0-index/index.pkl"

def ensure_faiss_index():
    os.makedirs(FAISS_FOLDER, exist_ok=True)

    if not os.path.exists("index_faiss/index.faiss"):
        urllib.request.urlretrieve(
            FAISS_URL,
            "index_faiss/index.faiss"
        )

    if not os.path.exists("index_faiss/index.pkl"):
        urllib.request.urlretrieve(
            PKL_URL,
            "index_faiss/index.pkl"
        )

ensure_faiss_index()

vector_store = FAISS.load_local(
    "index_faiss",
    embeddings,
    allow_dangerous_deserialization=True
)
chunk_lookup = {}
for document in vector_store.docstore._dict.values():
    conversation_id = document.metadata["conversation_id"]
    chunk_index = document.metadata["chunk_index"]
    chunk_lookup[(conversation_id, chunk_index)] = document

print("Vector store loaded successfully.")
def get_best_chunk(query: str):
    results = vector_store.similarity_search(
    query,
    k=1
)
    best_chunk = results[0]
    print(best_chunk.page_content)
    print(best_chunk.metadata)
    return best_chunk



print("Best chunk retrieved successfully.")
def get_neighboring_chunks(best_chunk):
    conversation_id = best_chunk.metadata["conversation_id"]
    chunk_index = best_chunk.metadata["chunk_index"]
    neighbor_indexes = [
    chunk_index - 1,
    chunk_index,
    chunk_index + 1
                    ]
    neighbors = []
    for index in neighbor_indexes:
        key = (conversation_id, index)
        if key in chunk_lookup:
            neighbors.append(chunk_lookup[key])
    return neighbors

print("Neighboring chunks retrieved successfully.")

def build_context(neighbors):
    chunks = [doc.page_content for doc in neighbors]
    context = "\n\n".join(chunks)
    return context

print("Context built successfully.")
def retrieve_context(query: str):
    best_chuck=get_best_chunk(query)
    neighbors=get_neighboring_chunks(best_chuck)
    context=build_context(neighbors)
    return context


    