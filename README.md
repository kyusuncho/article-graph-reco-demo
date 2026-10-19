# 기사 추천 그래프 데모

기사 추천 PoC의 파이프라인을 합성 데이터 위에서 다시 돌린 데모입니다. 기사·인물·브랜드·단체는 모두 LLM이 만든 가상 데이터입니다.

**데모:** https://kyusuncho.github.io/article-graph-reco-demo/

- 왼쪽: 기사 그래프. 비슷한 기사끼리 모여 있고, 노드를 누르면 기사 본문과 facet을 볼 수 있습니다.
- 오른쪽: 관심 분야·관심 tag·user history를 정하고 **추천하기**를 누르면 추천 기사와 그 근거가 나옵니다.

## 작동 방식

**색인 파이프라인** (기사가 들어올 때 한 번, 오프라인)

1. LLM이 기사에서 raw facet(topic, entity, activity, need, life context 등)을 추출합니다.
2. raw facet을 canonical tag와 닫힌 목록으로 normalize합니다 (key fold, alias 사전, LLM 동의어 리뷰).
3. 기사 embedding을 만듭니다.
4. 기사–facet 그래프와 기사 지도(Leiden community, t-SNE 배치)를 만듭니다.

**추천 파이프라인** (추천하기를 누를 때, 브라우저에서)

1. history와 관심 tag에서 출발하는 Forward Push(Personalized PageRank 근사)로 그래프 채널 점수를 냅니다.
2. profile embedding 유사도와 관심 분야 일치를 함께 가중합합니다.
3. MMR로 다양성을 맞추고 추천 근거(공유 facet, 유사한 history)를 붙입니다.

모든 추천 계산은 서버 없이 브라우저에서 돕니다.

## 로컬에서 보기

```bash
python3 -m http.server 8000
# http://localhost:8000
```

`index.html`을 파일로 바로 열면 데이터를 불러오지 못하므로 정적 서버로 여세요.

## 구성

| 경로 | 내용 |
|---|---|
| `index.html`, `app.js`, `style.css` | 데모 페이지 |
| `data/meta.json` | 카테고리, theme, facet, tag 정보 |
| `data/graph.json` | 기사 제목·위치·facet과 그래프 edge |
| `data/details.json` | 기사 부제와 본문, raw facet (지연 로딩) |
| `data/emb128.i8` | 브라우저용 기사 embedding |
| `vendor/` | [sigma.js](https://www.sigmajs.org) 3.0.0, [graphology](https://graphology.github.io) 0.25.4 (MIT) |
