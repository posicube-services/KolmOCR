<!-- bbox: [57,38,436,64] -->
# LLM 한국어-영어 통합 방법 실험

<!-- bbox: [77,100,187,118] -->
## 1) 목표 · 설정

<!-- bbox: [99,125,468,139] -->
- 한/영 동일 의미 입력에 대해 **동일(동등) 응답** 비율
              향상

<!-- bbox: [99,142,386,157] -->
- 영어 벤치마크 성능 유지 + 한국어 성능 향상

<!-- bbox: [99,161,468,175] -->
- 평가: arc_challenge ↔ ko_arc_challenge (1:1 번역쌍)

<!-- bbox: [77,178,129,193] -->
Stage 1

<!-- bbox: [77,197,156,212] -->
Contrastive

<!-- bbox: [77,217,129,232] -->
Stage 2

<!-- bbox: [77,237,140,252] -->
LM Head

<!-- bbox: [77,256,129,271] -->
Stage 3

<!-- bbox: [77,276,122,291] -->
Full FT

<!-- bbox: [77,303,235,318] -->
### 데이터 (Stage 1 병렬)

<!-- bbox: [77,323,468,402] -->
<table>
<thead>
<tr>
<th style="width:38%;">Dataset</th>
<th style="width:31%;">Train</th>
<th style="width:31%;">Val</th>
</tr>
</thead>
<tbody>
<tr>
<td>AIhub K2E (5종)</td>
<td>5,729,826</td>
<td>516,150</td>
</tr>
<tr>
<td><strong>Total</strong></td>
<td><strong>5.73M</strong></td>
<td><strong>0.52M</strong></td>
</tr>
</tbody>
</table>

<!-- bbox: [77,413,150,429] -->
### Baselines

<!-- bbox: [99,433,364,448] -->
- Sheared-LLaMA-1.3B-ShareGPT (base)
<!-- bbox: [99,451,296,466] -->
- polyglot-ko-1.3b (ko strong)

<!-- bbox: [99,433,364,448] -->
- Sheared-LLaMA-1.3B-ShareGPT (base)

<!-- bbox: [99,451,296,466] -->
- polyglot-ko-1.3b (ko strong)

<!-- bbox: [532,100,631,118] -->
## 2) 결과 요약

<!-- bbox: [532,125,922,295] -->
<table>
<thead>
<tr>
<th>Model</th>
<th style=" width:18%;">L1 일치</th>
<th style="width:16%;">L1 cs</th>
<th style="width:18%;">L17 일치</th>
<th style="width:16%;">L17 cs</th>
</tr>
</thead>
<tbody>
<tr>
<td>Base</td>
<td>40.02</td>
<td>0.99</td>
<td>40.02</td>
<td>0.93</td>
</tr>
<tr>
<td>ckpt-200</td>
<td>34.47</td>
<td>0.98</td>
<td>35.40</td>
<td>0.93</td>
</tr>
<tr>
<td>ckpt-400</td>
<td>36.09</td>
<td>0.98</td>
<td>34.90</td>
<td>0.81</td>
</tr>
<tr>
<td><strong>ckpt-600</strong></td>
<td><strong>36.26</strong></td>
<td>0.96</td>
<td><strong>35.49</strong></td>
<td>0.77</td>
</tr>
<tr>
<td>ckpt-800</td>
<td>35.24</td>
<td>0.93</td>
<td>33.70</td>
<td>0.74</td>
</tr>
</tbody>
</table>

<!-- bbox: [532,309,598,324] -->
### 대표 플롯

<!-- bbox: [533,331,719,486] -->
![cosine similarity (val)]( imgs/exp1_1.png)

<!-- bbox: [533,486,719,531] -->
*한/영 representation cosine similarity
              (validation)*

<!-- bbox: [735,331,921,486] -->
![training curve]( imgs/exp1_15.png)

<!-- bbox: [735,486,921,520] -->
*학습 곡선(대표 loss) – 장기 학습 시 추세 확인*

<!-- bbox: [532,546,643,561] -->
### 결론 · 다음 액션

<!-- bbox: [553,566,922,594] -->
- CL 단독은 일치도 개선에 유리, generation 결합 시 효과가 상쇄될 수
              있음
<!-- bbox: [553,598,922,627] -->
- 중간 layer(15~17)에서 의미 표현이 강해지는 패턴이 관찰됨
<!-- bbox: [553,630,922,659] -->
- 다음: **CL-only vs Gen-only vs
                CL+Gen** 대조 실험 + 스케줄/가중치 탐색

<!-- bbox: [553,566,922,594] -->
- CL 단독은 일치도 개선에 유리, generation 결합 시 효과가 상쇄될 수
              있음

<!-- bbox: [553,598,922,627] -->
- 중간 layer(15~17)에서 의미 표현이 강해지는 패턴이 관찰됨

<!-- bbox: [553,630,922,659] -->
- 다음: **CL-only vs Gen-only vs
                CL+Gen** 대조 실험 + 스케줄/가중치 탐색

<!-- bbox: [77,100,187,118] -->
## 1) 목표 · 설정

<!-- bbox: [77,303,235,318] -->
### 데이터 (Stage 1 병렬)

<!-- bbox: [77,323,468,402] -->
<table>
<thead>
<tr>
<th style="width:38%;">Dataset</th>
<th style="width:31%;">Train</th>
<th style="width:31%;">Val</th>
</tr>
</thead>
<tbody>
<tr>
<td>AIhub K2E (5종)</td>
<td>5,729,826</td>
<td>516,150</td>
</tr>
<tr>
<td><strong>Total</strong></td>
<td><strong>5.73M</strong></td>
<td><strong>0.52M</strong></td>
</tr>
</tbody>
</table>

<!-- bbox: [77,413,150,429] -->
### Baselines

<!-- bbox: [532,100,631,118] -->
## 2) 결과 요약

<!-- bbox: [532,125,922,295] -->
<table>
<thead>
<tr>
<th>Model</th>
<th style=" width:18%;">L1 일치</th>
<th style="width:16%;">L1 cs</th>
<th style="width:18%;">L17 일치</th>
<th style="width:16%;">L17 cs</th>
</tr>
</thead>
<tbody>
<tr>
<td>Base</td>
<td>40.02</td>
<td>0.99</td>
<td>40.02</td>
<td>0.93</td>
</tr>
<tr>
<td>ckpt-200</td>
<td>34.47</td>
<td>0.98</td>
<td>35.40</td>
<td>0.93</td>
</tr>
<tr>
<td>ckpt-400</td>
<td>36.09</td>
<td>0.98</td>
<td>34.90</td>
<td>0.81</td>
</tr>
<tr>
<td><strong>ckpt-600</strong></td>
<td><strong>36.26</strong></td>
<td>0.96</td>
<td><strong>35.49</strong></td>
<td>0.77</td>
</tr>
<tr>
<td>ckpt-800</td>
<td>35.24</td>
<td>0.93</td>
<td>33.70</td>
<td>0.74</td>
</tr>
</tbody>
</table>

<!-- bbox: [532,309,598,324] -->
### 대표 플롯

<!-- bbox: [532,546,643,561] -->
### 결론 · 다음 액션