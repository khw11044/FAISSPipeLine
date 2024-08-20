config = {
    "llm_predictor": {
        "model_name": "llama3.1",       # "llama3.1"
        "temperature": 0
    },
    "embed_model": "intfloat/multilingual-e5-large",  
    
    "text_split":{
        'chunk_size': 512,
        'chunk_overlap': 32
    },
    
    "save_data_path": './data',
    
    
    "search_type": "mmr",           # "mmr"  similarity   similarity_score_threshold
    'search_kwargs_k': 3,           # 유사도 기반 상위 k개
    'search_kwargs_lambda': 0.5,    # 0~1, 0에 가까울수록 다양성, 1에 가까울수록 관련성
    'score_threshold': 0.4,         # 0~1, 쿼리 문장과 최소한 0.x 이상의 유사도를 가진 문서만
    
    'bm25_k' : 3,                   # 검색어 기반 상위 k개
    "ensemble_search_type": "mmr",  # 앙상블 서칭 타입
    "ensemble_weight": [0.5,0.5],   # bm25와 vector 가중치
    


}