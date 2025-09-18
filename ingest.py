import os
import argparse
import json
import uuid
from typing import List, Dict, Any
from llama_index.core import Document
from tqdm import tqdm
import numpy as np
from collections import Counter
from sklearn.preprocessing import normalize



# 	lexical_weights (BGE-M3) không hiệu quả với văn bản rất ngắn hoặc ID sản phẩm nên phải dùng BM25.

# Milvus
from pymilvus import (
    connections, utility, Collection, CollectionSchema,
    FieldSchema, DataType, Function, FunctionType
)

# Embedding model (dense + sparse) — BGEM3
from FlagEmbedding import BGEM3FlagModel


# -------------------------
# Helpers
# -------------------------
def load_documents_from_json(json_path: str) -> List[Document]:
    print(f"Bắt đầu đọc dữ liệu từ file: {json_path}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file tại '{json_path}'")
        return []
    except json.JSONDecodeError:
        print(f"Lỗi: File '{json_path}' không phải là một file JSON hợp lệ.")
        return []

    documents: List[Document] = []
    for item in data:
        raw_content = item.get('content', '')
        metadata = item.get('metadata', {}) or {}
        
        content = f"{raw_content}\n{metadata}"
        
        documents.append(
            Document(
                text=content,
                metadata=metadata
            )
        )
    print(f"Đã tạo {len(documents)} Document.")
    return documents





# -------------------------
# Milvus ops
# -------------------------
def create_or_recreate_collection(collection_name: str, drop_existing: bool = False) -> Collection:
    if drop_existing and utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    if not utility.has_collection(collection_name):
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=32768, enable_analyzer=True),
            FieldSchema(name="dense", dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name="sparse", dtype=DataType.SPARSE_FLOAT_VECTOR,description="BM25 sparse vector"),
            
        ]
        
        # Tạo BM25 function
        bm25_function = Function(
            name="bm25_function",
            function_type=FunctionType.BM25,
            input_field_names=["text"],  # Input từ trường text
            output_field_names=["sparse"],  # Output ra trường sparse
            params={}
        )
        
        schema = CollectionSchema(fields, description="Hybrid search collection", enable_dynamic_field=False, functions=[bm25_function])
        col = Collection(collection_name, schema=schema)

        col.create_index(
            field_name="dense",
            index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 1024}},
        )
        col.create_index(
            field_name="sparse",
            index_params={"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "BM25"},
        )
    else:
        col = Collection(collection_name)

    col.load()
    return col


def insert_batches(col: Collection, rows: List[Dict[str, Any]], batch_size: int = 128):
    """
    rows: mỗi phần tử cần các khóa: id, doc_id, text, dense, sparse
    """
    n = len(rows)
    for i in tqdm(range(0, n, batch_size), desc="Insert to Milvus"):
        batch = rows[i:i+batch_size]
        ids = [r["id"] for r in batch]
        doc_ids = [r["doc_id"] for r in batch]
        texts = [r["text"] for r in batch]
        dense = [r["dense"] for r in batch]
        _ = col.insert([ids, doc_ids, texts, dense])
    col.flush()


# -------------------------
# Main ingest pipeline
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--milvus-uri", type=str, default=os.getenv("MILVUS_URI", "http://localhost:19530"))
    parser.add_argument("--collection", type=str, required=True)
    parser.add_argument("--input-json", type=str, required=True, help="Path to final_output.json-like file")
    parser.add_argument("--batch-size", type=int, default=128, help="Insert batch size for Milvus")
    parser.add_argument("--drop-existing", action="store_true")
    parser.add_argument("--normalize", action="store_true", help="L2 normalize dense vectors")
    args = parser.parse_args()


    connections.connect(alias="default", uri=args.milvus_uri)
    print(f"Connected to Milvus at {args.milvus_uri}")

    col = create_or_recreate_collection(args.collection, drop_existing=args.drop_existing)
    print(f"Collection ready: {args.collection}")
    
    model_name = "BAAI/bge-m3"
    print(f"Loading model: {model_name}")
    model = BGEM3FlagModel(model_name, use_fp16=True)
    
    docs = load_documents_from_json(args.input_json)
    print(f"Loaded {len(docs)} docs from {args.input_json}")
    
    doc_texts = [doc.get_content() for doc in docs]

    # --- 4. Tạo Dense Vectors bằng BGE-M3 trên toàn bộ tài liệu ---


    print("Đang tính toán dense vectors bằng BGE-M3...")
    
    enc = model.encode(doc_texts, return_dense=True, return_sparse=False, batch_size=32)
    dense_vectors = enc.get("dense_vecs")

    if args.normalize and dense_vectors is not None:
        print("Đang chuẩn hóa dense vectors...")
        dense_vectors = normalize(dense_vectors)


    # --- 5. Chuẩn bị và Insert vào Milvus ---
    if dense_vectors is None:
        raise RuntimeError("Không thể tạo dense vectors từ BGE-M3.")

    print("Đang chuẩn bị dữ liệu để insert...")
    rows = []
    for i, doc in enumerate(docs):
        rows.append({
            "id": uuid.uuid4().hex,
            "doc_id": doc.metadata.get("doc_id", f"doc_{i}"),
            "text": doc.get_content(),
            "dense": dense_vectors[i].tolist(),
        })

    insert_batches(col, rows, batch_size=args.batch_size)
    
    print(f"\nHoàn tất! Đã ingest {len(rows)} tài liệu vào collection '{args.collection}'.")
    print(f"Số lượng thực thể trong collection: {col.num_entities}")

if __name__ == "__main__":
    main()
