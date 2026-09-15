import os
import pickle

import faiss

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType,
)




load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "grounded_chunks"

FAISS_INDEX_PATH = "index_faiss/index.faiss"
FAISS_PKL_PATH = "index_faiss/index.pkl"

BATCH_SIZE = 512




print("Connecting to Qdrant...")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

print("Connected.")




if client.collection_exists(COLLECTION_NAME):
    print("Deleting old partial collection...")

    client.delete_collection(
        collection_name=COLLECTION_NAME
    )

    print("Old collection deleted.")




client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

print("Fresh collection created.")




client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="conversation_id",
    field_schema=PayloadSchemaType.KEYWORD
)

client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="chunk_index",
    field_schema=PayloadSchemaType.INTEGER
)

print("Payload indexes created.")



print("Loading FAISS index...")

faiss_index = faiss.read_index(
    FAISS_INDEX_PATH
)

print(
    f"FAISS contains {faiss_index.ntotal} vectors."
)



print("Loading FAISS document metadata...")

with open(FAISS_PKL_PATH, "rb") as file:
    docstore, index_to_docstore_id = pickle.load(file)

print(
    f"Loaded {len(index_to_docstore_id)} document mappings."
)



total = faiss_index.ntotal

print("\nStarting migration...\n")


for start in range(0, total, BATCH_SIZE):

    end = min(
        start + BATCH_SIZE,
        total
    )

    points = []


    for faiss_position in range(start, end):

    

        vector = faiss_index.reconstruct(
            faiss_position
        )


     

        document_id = index_to_docstore_id[
            faiss_position
        ]

        document = docstore._dict[
            document_id
        ]



        point = PointStruct(
            id=faiss_position,

            vector=vector.tolist(),

            payload={
                "text": document.page_content,

                "conversation_id":
                    document.metadata["conversation_id"],

                "chunk_index":
                    document.metadata["chunk_index"]
            }
        )

        points.append(point)



    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True
    )

    print(
        f"Uploaded {end}/{total} chunks"
    )




info = client.get_collection(
    collection_name=COLLECTION_NAME
)

print("\nMigration finished.")
print(
    f"Qdrant points: {info.points_count}"
)