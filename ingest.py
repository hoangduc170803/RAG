# ingest.py
import os
import argparse
import json
import uuid
from typing import List, Dict, Any
from llama_index.core import Document
from tqdm import tqdm
from sklearn.preprocessing import normalize
import re

from pymilvus import (
    connections, utility, Collection, CollectionSchema,
    FieldSchema, DataType, Function, FunctionType
)

from FlagEmbedding import BGEM3FlagModel


# -------------------------
# Helpers (format hiển thị)
# -------------------------
def pretty_kv_block(raw_content: str) -> str:
    """
    Chuyển chuỗi dạng list KV:
      "[A:1, B:2, Phí:200,000, C:3]"
    -> xuống dòng theo cặp "key: value", KHÔNG tách dấu phẩy trong số tiền.
    """
    if not raw_content:
        return ""
    inner = raw_content.strip()
    if inner.startswith("[") and inner.endswith("]"):
        inner = inner[1:-1].strip()
    if not inner:
        return ""
    parts = re.split(r", (?=[^,\]]+?:)", inner)
    parts = [p.strip() for p in parts if p.strip()]
    # Loại khoảng trắng đầu dòng (ví dụ " Cước...")
    parts = [re.sub(r"^\s+", "", p) for p in parts]
    return "\n".join(parts)


def build_text_with_source(raw_content: str, metadata: dict) -> str:
    src   = metadata.get("source", "") or ""
    sheet = metadata.get("sheet", "") or ""

    header_parts = []
    if src:
        header_parts.append(f"NGUỒN: {src}")
    if sheet and "Sheet:" not in src:
        header_parts.append(f"Sheet: {sheet}")
    header = " | ".join(header_parts)

    body = pretty_kv_block(raw_content)
    return (header + ("\n" if header and body else "") + body).strip()


def load_documents_from_json(json_path: str) -> List[Document]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents: List[Document] = []
    for item in data:
        raw_content = item.get('content', '') or ''
        metadata    = item.get('metadata', {}) or {}

        # Cho phép cài doc_id từ metadata, fallback theo vị trí
        # (llama_index Document không có doc_id riêng, mình giữ trong metadata)
        content = build_text_with_source(raw_content, metadata)
        documents.append(Document(text=content, metadata=metadata))
    print(f"Đã tạo {len(documents)} Document.")
    return documents


# -------------------------
# Milvus
# -------------------------
def create_or_recreate_collection(collection_name: str, drop_existing: bool = False) -> Collection:
    """
    - Field text: enable_analyzer=True với ICU + lowercase (+ stopwords nếu bạn mount file)
    - Function BM25: KHÔNG có params
    - BM25 params (k1,b) đặt trong create_index của sparse
    """
    if drop_existing and utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"Dropped existing collection: {collection_name}")

    if not utility.has_collection(collection_name):
        fields = [
            FieldSchema(name="id",      dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="doc_id",  dtype=DataType.VARCHAR, max_length=128),
            # Nếu bạn muốn multi-language, thêm Field 'lang' và cập nhật analyzer_params (by_field).
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=65535,
                enable_analyzer=True,
                analyzer_params={
                    "tokenizer": {"type": "icu", "options": {"locale": "vi"}}
                }
            ),
            FieldSchema(name="source",  dtype=DataType.VARCHAR, max_length=32768),
            FieldSchema(name="dense",   dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name="sparse",  dtype=DataType.SPARSE_FLOAT_VECTOR, description="BM25 sparse vector"),
        ]

        bm25_function = Function(
            name="bm25_function",
            function_type=FunctionType.BM25,
            input_field_names=["text"],     # BM25 lấy input từ field text
            output_field_names=["sparse"],  # ghi kết quả ra field sparse
            params={}                       # KHÔNG đặt k1/b ở đây (Milvus không nhận)
        )

        schema = CollectionSchema(
            fields=fields,
            description="Hybrid search collection (VN ICU analyzer, BM25 sparse)",
            enable_dynamic_field=False,
            functions=[bm25_function],
        )
        col = Collection(collection_name, schema=schema)

        # Index cho dense
        col.create_index(
            field_name="dense",
            index_params={
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "params": {"nlist": 1024},
            },
        )

        # Index cho sparse (đặt bm25_k1/b ở đây)
        col.create_index(
            field_name="sparse",
            index_params={
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "BM25",
                "params": {"bm25_k1": 1.4, "bm25_b": 0.65},
            },
        )
        print(f"Created new collection: {collection_name}")
    else:
        col = Collection(collection_name)
        print(f"Using existing collection: {collection_name}")

    col.load()
    return col


def insert_batches(col: Collection, rows: List[Dict[str, Any]], batch_size: int = 128):
    """
    Thứ tự mảng insert PHẢI trùng thứ tự field trong schema:
      id, doc_id, text, source, dense, (sparse là function output, KHÔNG insert)
    """
    n = len(rows)
    for i in tqdm(range(0, n, batch_size), desc="Insert to Milvus"):
        batch = rows[i:i+batch_size]
        ids     = [r["id"] for r in batch]
        doc_ids = [r["doc_id"] for r in batch]
        texts   = [r["text"] for r in batch]
        sources = [r.get("source", "") for r in batch]
        dense   = [r["dense"] for r in batch]

        # CHÚ Ý: danh sách data phải có 5 list tương ứng 5 field (trừ sparse)
        col.insert([ids, doc_ids, texts, sources, dense])
    col.flush()


# -------------------------
# Main
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

    # Connect
    connections.connect(alias="default", uri=args.milvus_uri)
    print(f"Connected to Milvus at {args.milvus_uri}")

    # Create/Load collection
    col = create_or_recreate_collection(args.collection, drop_existing=args.drop_existing)
    print(f"Collection ready: {args.collection}")

    # Load data
    docs = load_documents_from_json(args.input_json)
    print(f"Loaded {len(docs)} docs from {args.input_json}")

    # Build list text + doc_id + source
    doc_texts = [doc.text for doc in docs]
    doc_ids = [doc.metadata.get("doc_id", f"doc_{i+1}") for i, doc in enumerate(docs)]
    sources   = [doc.metadata.get("source", "") for doc in docs]

    # Embedding
    print("Loading embedding model: BAAI/bge-m3")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    print("Encoding dense vectors...")
    enc = model.encode(doc_texts, return_dense=True, return_sparse=False, batch_size=32)
    dense_vectors = enc.get("dense_vecs")
    if dense_vectors is None:
        raise RuntimeError("Không thể tạo dense vectors từ BGE-M3.")

    if args.normalize:
        print("L2-normalize dense vectors...")
        dense_vectors = normalize(dense_vectors)

    # Chuẩn bị rows
    rows = []
    for i, (text, doc_id, source) in enumerate(zip(doc_texts, doc_ids, sources)):
        rows.append({
            "id": uuid.uuid4().hex,
            "doc_id": doc_id,
            "text": text,                      # Analyzer ICU + filters sẽ áp cho field này (BM25 dùng)
            "source": source,                  # Để display/trace
            "dense": dense_vectors[i].tolist()
        })

    print(f"Inserting {len(rows)} rows ...")
    insert_batches(col, rows, batch_size=args.batch_size)

    print(f"\nHoàn tất! Đã ingest {len(rows)} tài liệu vào collection '{args.collection}'.")
    print(f"Số lượng thực thể trong collection: {col.num_entities}")


if __name__ == "__main__":
    main()
