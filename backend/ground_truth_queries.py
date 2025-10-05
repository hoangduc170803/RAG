"""
Ground truth queries và expected results cho RAG evaluation.
Dựa trên dữ liệu từ final_output.json với doc_id format: doc_1, doc_2, ...
"""

from typing import List, Dict, Any

# Định nghĩa các queries với expected results
# doc_id tương ứng với index trong final_output.json: index 0 = doc_1, index 1 = doc_2, etc.
"""
Ground truth queries và expected results cho RAG evaluation.
Dựa trên dữ liệu từ final_output.json với doc_id format: doc_1, doc_2, ...
"""

GROUND_TRUTH_QUERIES = [
    # === QUERIES VỀ GÓI CƯỚC ===
    {
        "query": "Mã KM N200X có phí tham gia bao nhiêu?",
        "expected_keywords": ["N200X", "phí tham gia", "200,000", "200000"],
        "expected_doc_ids": ["doc_11", "doc_17", "doc_23", "doc_29", "doc_35", "doc_41", "doc_47", "doc_53", "doc_59", "doc_65"],
        "expected_sources": ["PYC-17290_PYC khai bao PBH goi NxxX_tables.xlsx"],
        "category": "promotion_query"
    },
    {
        "query": "Mã KM N250X có phí tham gia bao nhiêu?",
        "expected_keywords": ["N250X", "phí tham gia", "250,000", "250000"],
        "expected_doc_ids": ["doc_12", "doc_18", "doc_24", "doc_30", "doc_36", "doc_42", "doc_48", "doc_54", "doc_60", "doc_66"],
        "expected_sources": ["PYC-17290_PYC khai bao PBH goi NxxX_tables.xlsx"],
        "category": "promotion_query"
    },
    {
        "query": "Mã KM N300X có phí tham gia bao nhiêu?",
        "expected_keywords": ["N300X", "phí tham gia", "300,000", "300000"],
        "expected_doc_ids": ["doc_13", "doc_19", "doc_25", "doc_31", "doc_37", "doc_43", "doc_49", "doc_55", "doc_61", "doc_67"],
        "expected_sources": ["PYC-17290_PYC khai bao PBH goi NxxX_tables.xlsx"],
        "category": "promotion_query"
    },
    {
        "query": "Mã KM N500X có tổng phí bán hàng là bao nhiêu?",
        "expected_keywords": ["N500X", "Tổng Phí bán hàng"],
        "expected_doc_ids": ["doc_14", "doc_20", "doc_26", "doc_32", "doc_38", "doc_44", "doc_50", "doc_56", "doc_62", "doc_68"],
        "expected_sources": ["PYC-17290_PYC khai bao PBH goi NxxX_tables.xlsx"],
        "category": "promotion_query"
    },
    {
        "query": "Mã KM N1000X có phí phát triển thuê bao là bao nhiêu?",
        "expected_keywords": ["N1000X", "Phí phát triển"],
        "expected_doc_ids": ["doc_15", "doc_21", "doc_27", "doc_33", "doc_39", "doc_45", "doc_51", "doc_57", "doc_63", "doc_69"],
        "expected_sources": ["PYC-17290_PYC khai bao PBH goi NxxX_tables.xlsx"],
        "category": "promotion_query"
    },

    # === QUERIES VỀ GÓI CƯỚC ===
    {
        "query": "Gói TOUR có những mã lý do nào?",
        "expected_keywords": ["TOUR", "Mã lý do", "TOUR_DL60", "TOUR_DL120", "TOUR_QR"],
        "expected_doc_ids": ["doc_3", "doc_4", "doc_9"],
        "expected_sources": ["4197906_PYC khai phí bán hàng gói M2M7S1, SD70TS, TOUR, M2M10_128S_tables.xlsx"],
        "category": "code_query"
    },
    {
        "query": "Gói cước M2M7S1 có phí tham gia là bao nhiêu?",
        "expected_keywords": ["M2M7S1", "Phí tham gia gói"],
        "expected_doc_ids": ["doc_1"],
        "expected_sources": ["4197906_PYC khai phí bán hàng gói M2M7S1, SD70TS, TOUR, M2M10_128S_tables.xlsx"],
        "category": "fee_query"
    },
    {
        "query": "Phí bán hàng của gói SD70TS là bao nhiêu?",
        "expected_keywords": ["SD70TS", "Phí bán hàng"],
        "expected_doc_ids": ["doc_2"],
        "expected_sources": ["4197906_PYC khai phí bán hàng gói M2M7S1, SD70TS, TOUR, M2M10_128S_tables.xlsx"],
        "category": "fee_query"
    },
    {
        "query": "Gói M2M10_128S thuộc dịch vụ nào?",
        "expected_keywords": ["M2M10_128S", "Dcom trả sau"],
        "expected_doc_ids": ["doc_5", "doc_6", "doc_7", "doc_8"],
        "expected_sources": ["4197906_PYC khai phí bán hàng gói M2M7S1, SD70TS, TOUR, M2M10_128S_tables.xlsx"],
        "category": "service_query"
    },

    # === QUERY SO SÁNH ===
    {
        "query": "Gói cước M2M8_100SMS có những chính sách phí bán hàng nào?",
        "expected_keywords": ["M2M8_100SMS", "Phí bán hàng", "Tổng", "PTTB"],
        "expected_doc_ids": ["doc_71", "doc_72", "doc_73"],
        "expected_sources": ["PYC-24130_ PYC khai bao PBH dau noi goi dau noi goi Dcom, M2M thang 7.2024_tables.xlsx"],
        "category": "fee_policy_query"
    },
]

def convert_index_to_doc_id(index: int) -> str:
    """Convert index (0-based) to doc_id format (doc_1, doc_2, ...)"""
    return f"doc_{index + 1}"

def convert_doc_id_to_index(doc_id: str) -> int:
    """Convert doc_id format to index"""
    return int(doc_id.replace("doc_", "")) - 1

def get_queries_by_category(category: str) -> List[Dict[str, Any]]:
    """Lấy queries theo category"""
    return [q for q in GROUND_TRUTH_QUERIES if q["category"] == category]

def get_all_categories() -> List[str]:
    """Lấy danh sách tất cả categories"""
    return list(set(q["category"] for q in GROUND_TRUTH_QUERIES))

def get_query_statistics() -> Dict[str, int]:
    """Thống kê queries theo category"""
    stats = {}
    for category in get_all_categories():
        stats[category] = len(get_queries_by_category(category))
    stats["total"] = len(GROUND_TRUTH_QUERIES)
    return stats