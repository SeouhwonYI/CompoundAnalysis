# CompoundAnalysis

`CompoundAnalysis`는 레시피-분자 연결 데이터를 생성하고(`generate`), cuisine별 분자 클러스터 분석/시각화를 수행하는(`analysis`) 파이프라인입니다.

## 폴더 구조
- `generate/generate.py`: cuisine별 그래프 입력 CSV 생성
- `analysis/cuisine_only_cluster_analysis.py`: cuisine별 클러스터 분석 + 시각화
- `preprocess/`: 전처리 파일
- `result/`: 생성 결과 저장 경로

## 실행 환경
- Python 3.10+ 권장

## 패키지 설치
프로젝트 루트(`CompoundAnalysis`)에서 실행:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 파이프라인 실행 순서
1. `generate/generate.py` 실행
2. `analysis/cuisine_only_cluster_analysis.py` 실행

---

## 1) Generate Stage
### 스크립트
- `generate/generate.py`

### 주요 출력
- `result/recipes_long_normalized.csv`
- `result/graph/recipe_molecule_edges.csv`
- `result/graph/recipe_unk_mass.csv`
- `result/graph/<cuisine>/000_recipe_molecule_edges.csv`
- `result/graph/<cuisine>/001_molecule_weight.csv`
- `result/graph/<cuisine>/002_molecule_recipe_edges.csv`
- `result/graph/ALL/*` (`ALL` export 활성 시)
- `result/analysis/analysis_summary.csv`

### cuisine 동작
- `--cuisines` 생략: 전체 cuisine + `ALL` 생성
- `--cuisines` 지정: 지정 cuisine만 생성
- `--include-all`: 지정 cuisine + `ALL` 함께 생성

### 실행 예시
전체 cuisine + ALL:
```bash
python generate/generate.py
```

특정 cuisine만:
```bash
python generate/generate.py --cuisines Korean Thai
```

특정 cuisine + ALL:
```bash
python generate/generate.py --cuisines Korean Thai --include-all
```

(선택) 클러스터 개요 플롯까지 생성:
```bash
python generate/generate.py --export-cluster-overview-plots
```

---

## 2) Analysis Stage
### 스크립트
- `analysis/cuisine_only_cluster_analysis.py`

### 입력 우선순위
1. `saved_state/*` (있으면 우선 사용)
2. `result/graph/<cuisine>/000_recipe_molecule_edges.csv`
3. `result/recipes_long_normalized.csv`의 ingredient fallback

### 주요 출력
- `result/plots/<cuisine>/<cuisine>_summary.png`
- `result/plots/<cuisine>/<cuisine>_moleculespace_graph.png`
- `result/plots/<cuisine>/<cuisine>_moleculespace_graph.html`
- `result/analysis/<cuisine>/000_cluster_summary.csv`

### 실행 예시
`result/graph`에서 cuisine 자동 탐색:
```bash
python analysis/cuisine_only_cluster_analysis.py
```

특정 cuisine 분석:
```bash
python analysis/cuisine_only_cluster_analysis.py --cuisines Korean Thai
```

전체 cuisine 분석:
```bash
python analysis/cuisine_only_cluster_analysis.py --all-cuisines
```

## 빠른 시작 (복사해서 실행)
```bash
cd CompoundAnalysis
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python generate/generate.py
python analysis/cuisine_only_cluster_analysis.py
```
