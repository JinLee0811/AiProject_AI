# AI

## Project Start


### .env

```
aws_access_key_id 
aws_secret_access_key 
endpoint_url 
```


### 터미널

```
# ai 폴더로 이동
cd ai

# 선택사항: (충돌 방지) 가상환경 설정 
conda create -n cropdoctor python=3.8
conda activate cropdoctor

# 필요 패키지 설치 
pip install -r requirements.txt

# 실행 
uvicorn main:app --reload
```
