# 1. 실험을 위해 조절할 것들 

### 1. 데이터셋 구축 부분 

- utils/utils.py 에서 process_pdfs_from_dataframe_faiss 함수의 chunks를 만들어내는 process_pdf를 바꿔가며 실험하자 

- pdf별로 다르게 chunk를 만들어내는 전략도 생각해볼만 하다. (어떤거는 표 위주, 어떤것은 내용 위주)

- config파일에서 embed_model을 바꿔가며 실험해보자.

- config파일에서 text_split부분에서 chunk_size를 바꿔가며 실험해보자 

### 2. LLM 부분 

- config 파일에서 model_name을 바꿔보자 

### 3. retriever 부분 

- 리트리버의 search_type를 바꿔가보며 실험해보자 
- 리트리버의 k를 바꿔가보며 실험해보자
- bm25의 k를 바꿔가보며 실험해보자 
- 앙상블을 사용할것인지 멀티쿼리를 사용할 것인지 
- 앙상블의 weight를 조절하자 


# 2. 수행 순서별 코드 

1. 1.get_testdata.ipynb을 통해 test데이터를 DB화 

2. 2.test.ipynb으로 submit.csv만들고 제출 

3. 1.get_traindata.ipynb은 train데이터로 db만들고 결과 확인 

# 3. colab에서 돌리고 싶다면 

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
# 해당 폴더채로 구글 드라이브에 넣고 불러오기 
%cd drive/MyDrive/2024/LLM_study

!ls
```

```python
# Ragpipeline 불러들여서 실험하기 
# RAGpipeline의 llm만 Huggingface로 바꿔도 된다.
from utils.RagPipeline import Ragpipeline
from config import config
```