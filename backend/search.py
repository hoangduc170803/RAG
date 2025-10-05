import requests
from pymilvus import Collection, connections
from pymilvus import AnnSearchRequest, WeightedRanker, RRFRanker


# Kết nối đến các service
MILVUS_HOST = "milvus"
MILVUS_PORT = "19530"
TEI_ENDPOINT = "http://tei:80/embed"  # Endpoint của TEI cho dense
COLLECTION_NAME = "my_rag_collection"


connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
collection = Collection(COLLECTION_NAME)
collection.load()
print("Đã kết nối Milvus và tải collection.")


def get_dense_embedding(query: str):
    """Lấy dense embedding từ TEI service."""
    try:
        res_dense = requests.post(TEI_ENDPOINT, json={"inputs": [query], "model": "BAAI/bge-m3"})
        res_dense.raise_for_status()
        dense_vec = res_dense.json()[0] 
        print("-> Lấy dense vector thành công.")
        return dense_vec
    except requests.RequestException as e:
        print(f"Lỗi khi gọi TEI service: {e}")
        return None


def search_dense_only(query: str, top_k: int = 5):
    """Tìm kiếm chỉ bằng dense vector (semantic search)."""
    print(f"Đang tìm kiếm semantic cho: '{query}'")
    
    dense_vec = get_dense_embedding(query)
    if not dense_vec:
        return []
    
    search_params = {"metric_type": "COSINE", "params": {}}
    results = collection.search(
        data=[dense_vec],
        anns_field="dense",
        param=search_params,
        limit=top_k,
        output_fields=["doc_id", "text"]
    )
    
    retrieved_docs = []
    if results and results[0]:
        for hit in results[0]:
            retrieved_docs.append({
                "text": hit.entity.get('text'),
                "score": hit.score,  
                "doc_id": hit.entity.get('doc_id'),
                "search_type": "dense"
            })
    return retrieved_docs


def search_bm25_only(query: str, top_k: int = 5):
    """Tìm kiếm chỉ bằng BM25 (lexical search)."""
    print(f"Đang tìm kiếm BM25 cho: '{query}'")
    
    search_params = {"metric_type": "BM25", "params": {}}
    results = collection.search(
        data=[query],  # Truyền query text trực tiếp
        anns_field="sparse",
        param=search_params,
        limit=top_k,
        output_fields=["doc_id", "text"]
    )
    
    retrieved_docs = []
    if results and results[0]:
        for hit in results[0]:
            retrieved_docs.append({
                "text": hit.entity.get('text'),
                "score": hit.score,
                "doc_id": hit.entity.get('doc_id'),
                "search_type": "bm25"
            })
    return retrieved_docs


def hybrid_search(query: str, top_k: int = 5):
    """Thực hiện tìm kiếm hybrid với RRF reranking."""
    print(f"Đang thực hiện hybrid search cho: '{query}'")
    
    # 1. Lấy dense vector từ TEI
    dense_vec = get_dense_embedding(query)
    if not dense_vec:
        print("Không thể lấy dense vector, fallback sang BM25 only")
        return search_bm25_only(query, top_k)
    
    # 2. Định nghĩa search requests cho hybrid search
    dense_request = AnnSearchRequest(
        data=[dense_vec],
        anns_field="dense",
        param={"metric_type": "COSINE", "params": {}},
        limit=top_k * 3  
    )
    bm25_request = AnnSearchRequest(
        data=[query],  # Truyền query text trực tiếp cho BM25
        anns_field="sparse", 
        param={"metric_type": "BM25", "params": {}},
        limit=top_k * 8
    )
    anchor_ranker = WeightedRanker(0.3, 0.7)
    anchor = collection.hybrid_search(
        reqs=[dense_request, bm25_request],
        rerank=anchor_ranker,
        limit=1,
        output_fields=["doc_id","text"]
    )
    final, seen = [], set()
    if anchor and anchor[0]:
        h = anchor[0][0]
        did = h.entity.get("doc_id")
        if did:
            final.append({"doc_id": did, "text": h.entity.get("text"), "score": h.score, "search_type":"anchor"})
            seen.add(did)        
    
    print(f"Đang tìm kiếm top {top_k} kết quả với hybrid search...")
    
    ranker = WeightedRanker(0.1, 0.9)
    
 

        # Sử dụng list của AnnSearchRequest objects
    fused = collection.hybrid_search(
        reqs=[dense_request, bm25_request],
        rerank=ranker,  
        limit=top_k,
        output_fields=["doc_id", "text"]
    )
    
    if fused and fused[0]:
        for h in fused[0]:
            did = h.entity.get("doc_id")
            if did and did not in seen:
                final.append({"doc_id": did, "text": h.entity.get("text"), "score": h.score, "search_type":"rrf"})
                seen.add(did)
            if len(final) >= top_k:
                break

        
    return final
        

    
    




def compare_search_methods(query: str, top_k: int = 3):
    """So sánh các phương pháp tìm kiếm khác nhau."""
    print(f"\n{'='*60}")
    print(f"SO SÁNH CÁC PHƯƠNG PHÁP TÌM KIẾM")
    print(f"Query: '{query}'")
    print(f"{'='*60}")
    
    # Dense search
    print(f"\n--- DENSE SEARCH (Semantic) ---")
    dense_results = search_dense_only(query, top_k)
    for i, result in enumerate(dense_results, 1):
        print(f"{i}. Score: {result['score']:.4f} | Doc: {result['doc_id']}")
        print(f"   Text: {result['text'][:100]}...")
    
    # BM25 search  
    print(f"\n--- BM25 SEARCH (Lexical) ---")
    bm25_results = search_bm25_only(query, top_k)
    for i, result in enumerate(bm25_results, 1):
        print(f"{i}. Score: {result['score']:.4f} | Doc: {result['doc_id']}")
        print(f"   Text: {result['text'][:100]}...")
    
    # Hybrid search
    print(f"\n--- HYBRID SEARCH (Dense + BM25) ---")
    hybrid_results = hybrid_search(query, top_k)
    for i, result in enumerate(hybrid_results, 1):
        print(f"{i}. Score: {result['score']:.4f} | Doc: {result['doc_id']}")
        print(f"   Text: {result['text'][:100]}...")


if __name__ == "__main__":
    query = "Mã KM N200X có phí tham gia bao nhiêu?"
    
    # Test basic hybrid search
    print("=== HYBRID SEARCH TEST ===")
    results = hybrid_search(query)
    for i, res in enumerate(results, 1):
        print(f"{i}. Score: {res['score']:.4f} | Type: {res['search_type']} | Doc: {res['doc_id']}")
        print(f"   Text: {res['text'][:150]}...")
        print()
    
    # Compare all methods
    compare_search_methods(query, top_k=10)
    
    