"""결과 페이지용 '모델 비교와 최종 모델 선정' 섹션 생성.

입력: outputs/tables/17_zoo_cv2025_육지.csv (zoo.py 결과, 2025 교차검증)
출력: drop_classifier/outputs/model_section.html
실행: python -m drop_classifier.src.model_section
"""
import html

import pandas as pd

from common.config import ROOT

TAB = ROOT / "drop_classifier" / "outputs" / "tables"
OUT = ROOT / "drop_classifier" / "outputs" / "model_section.html"

# 모델 이름 → (계열, 어떤 모델인가)
DESC = {
    "Ridge": ("선형", "요인마다 가중치를 곱해 더하는 직선형 모델. 가중치가 너무 커지지 않게 눌러 안정적"),
    "Lasso": ("선형", "Ridge와 같지만 쓸모없는 요인의 가중치를 0으로 만들어 요인을 골라냄"),
    "ElasticNet": ("선형", "Ridge와 Lasso를 섞은 선형 모델"),
    "Huber": ("선형", "튀는 값(이상치)의 영향을 줄이는 선형 회귀"),
    "로지스틱 회귀": ("선형", "급락할 확률을 직접 내는 가장 기본적인 분류 모델 (협의자료 예시)"),
    "의사결정나무(회귀)": ("트리", "'수요가 낮은가? → 맑은가?' 같은 질문을 이어 붙인 나무 하나 (협의자료 예시)"),
    "의사결정나무(분류)": ("트리", "같은 나무 하나로 급락 여부를 바로 분류 (협의자료 예시)"),
    "랜덤포레스트(회귀)": ("트리 묶음", "자료를 조금씩 바꿔 만든 의사결정나무 500개의 예측을 평균"),
    "랜덤포레스트(분류)": ("트리 묶음", "나무 500개의 다수결로 급락 여부를 분류"),
    "엑스트라트리": ("트리 묶음", "랜덤포레스트와 비슷하되 나무를 더 무작위로 쪼갬"),
    "GradientBoosting": ("부스팅", "앞 나무가 틀린 부분을 다음 나무가 고치도록 순서대로 쌓음"),
    "HistGB": ("부스팅", "구간을 나눠 빠르게 계산하는 부스팅"),
    "XGBoost(회귀)": ("부스팅", "대회에서 널리 쓰이는 부스팅. 과적합 방지 장치가 많음"),
    "XGBoost(분류)": ("부스팅", "XGBoost로 급락 여부를 분류"),
    "LightGBM(회귀)": ("부스팅", "빠르고 정확한 부스팅. 현업 SMP 예측 프로그램이 쓰는 모델"),
    "LightGBM(분류)": ("부스팅", "LightGBM으로 급락 여부를 분류"),
    "CatBoost(회귀)": ("부스팅", "범주형 값 처리가 강한 부스팅"),
    "CatBoost(분류)": ("부스팅", "CatBoost로 급락 여부를 분류"),
    "EBM(회귀)": ("설명형 모델", "요인별로 굽은 관계(예: 연휴 며칠째)를 따로 배워 더함. 각 요인의 영향을 그림으로 볼 수 있음"),
    "EBM(분류)": ("설명형 모델", "EBM으로 급락 여부를 분류"),
    "SVR": ("기타", "자료를 높은 차원으로 옮겨 곡선 관계를 찾는 서포트 벡터 회귀"),
    "KNN": ("기타", "과거에 조건이 가장 비슷했던 날 15개의 결과를 참고"),
    "MLP": ("딥러닝", "층을 쌓은 기본 신경망"),
    "TabPFN": ("기초모델", "수많은 가상 표 자료로 미리 학습한 표 데이터용 AI. 우리 자료로는 짧게만 맞춤"),
    "기준선(전날 한낮 최저)": ("기준선", "학습 없이 '어제 한낮이 낮았으면 오늘도 낮다'로 판단"),
    "Chronos-Bolt(제로샷)": ("기초모델", "아마존의 시계열 AI. 과거 SMP 512시간만 보고 내일을 예측 (우리 자료로 학습 안 함)"),
    "Chronos-2(제로샷)": ("기초모델", "Chronos의 최신판. 같은 방식"),
    "LSTM": ("딥러닝", "시간 순서를 기억하는 순환 신경망. 지난 7일 시간별 SMP + 요인"),
    "GRU": ("딥러닝", "LSTM을 단순화한 순환 신경망"),
    "1D-CNN": ("딥러닝", "시간별 SMP에서 반복 모양을 찾는 합성곱 신경망"),
    "Transformer": ("딥러닝", "ChatGPT와 같은 주의(attention) 구조. 하루를 한 단위로 봄"),
    "Ridge+Chronos피처": ("기초모델 결합", "Chronos 예측값을 입력 요인으로 추가한 Ridge"),
    "EBM+Chronos피처": ("기초모델 결합", "Chronos 예측값을 입력 요인으로 추가한 EBM"),
    "랜덤포레스트+Chronos피처": ("기초모델 결합", "Chronos 예측값을 입력 요인으로 추가한 랜덤포레스트"),
}


def describe(name, kind):
    if name in DESC:
        return DESC[name]
    if name.startswith("순위평균"):
        return "앙상블", "여러 모델의 '위험 순위'를 평균 (확률·금액을 내는 모델을 섞을 때)"
    if name.startswith("다 넣기"):
        return "앙상블", "좋은 모델·딥러닝·기초모델을 모두 평균"
    if name.startswith("Chronos피처"):
        return "앙상블", "Chronos 예측을 입력으로 넣은 모델들의 평균"
    return "앙상블", "이름에 적힌 모델들의 예측 금액을 평균"


def main():
    z = pd.read_csv(TAB / "17_zoo_cv2025_육지.csv")
    z = z.sort_values(["헛경보@90%", "헛경보@100%", "PR-AUC"], ascending=[True, True, False]).reset_index(drop=True)
    rows = []
    for i, r in z.iterrows():
        fam, desc = describe(r["모델"], r["구분"])
        final = r["모델"] == "Ridge+EBM+랜덤포레스트"
        cur = r["모델"] == "Ridge+EBM (현 모델)"
        name = html.escape(r["모델"].replace(" (현 모델)", ""))
        tag = (' <span class="pill">최종</span>' if final else ' <span class="pill soft">이전</span>' if cur else "")
        tag += ' <span class="pill soft">추가</span>' if "추가" in r["구분"] else ""
        rows.append(f'<tr{" class=total" if final else ""}><td class="n">{i+1}</td><td><b>{name}</b>{tag}</td>'
                    f'<td class="n">{int(r["헛경보@90%"])}</td><td class="n">{int(r["헛경보@100%"])}</td>'
                    f'<td class="n">{r["ROC-AUC"]:.3f}</td><td class="fam">{fam}</td><td class="desc">{html.escape(desc)}</td></tr>')
    sec = f"""
<section id="models">
  <h2>모델 비교와 최종 모델 선정</h2>
  <p>Chronos·TabPFN·LSTM·Transformer·XGBoost·LightGBM 등 널리 쓰이는 모델 {(~z['구분'].str.startswith('앙상블')).sum()}종과, 이들을 3개 이상 섞은 앙상블 {z['구분'].str.startswith('앙상블').sum()}종을 <b>같은 조건</b>에서 비교했습니다. 입력 정보는 모두 같고(전날 15시 공개 정보), 2024년으로 학습해 2025년 상반기를, 2024~2025년 상반기로 학습해 2025년 하반기를 맞히게 했습니다. 2026년은 고르는 데 쓰지 않았습니다.</p>
  <p>회사는 <b>헛경보가 많더라도 급락을 놓치지 않는 쪽</b>을 원했기 때문에, <b>2025년 급락의 90%를 잡으려면 헛경보가 며칠 필요한가</b>로 순위를 매겼습니다(적을수록 좋음). 교체는 이전 모델보다 15% 이상 좋을 때만 하기로 미리 정했습니다. "추가" 표시는 1차 결과를 본 뒤 요청으로 더한 후보입니다.</p>
  <div class="panel scroll" style="max-height:620px"><table class="models"><thead><tr><th class="n">순위</th><th>모델</th><th class="n">90% 잡을 때 헛경보</th><th class="n">100% 잡을 때 헛경보</th><th class="n">ROC-AUC</th><th>계열</th><th>어떤 모델인가</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>

  <h3>최종 모델: Ridge + EBM + 랜덤포레스트</h3>
  <p>세 모델이 각각 내일 한낮 최저 SMP를 예측하고, 그 평균이 문턱보다 낮으면 경보합니다.</p>
  <div class="factors">
    <div class="factor"><h3>Ridge</h3><p>요인마다 가중치를 곱해 더하는 직선형 모델. "최근 SMP가 낮을수록, 수요가 낮을수록 위험" 같은 <b>전체적인 경향</b>을 잡습니다.</p></div>
    <div class="factor"><h3>EBM</h3><p>요인별로 굽은 관계를 따로 배웁니다. "SMP가 110원 아래로 내려오면 위험이 급격히 커진다"처럼 <b>직선으로는 안 되는 모양</b>을 잡습니다.</p></div>
    <div class="factor"><h3>랜덤포레스트</h3><p>의사결정나무 500개의 평균. "휴일이면서, 맑고, 수요도 낮을 때"처럼 <b>조건이 겹칠 때만 생기는 위험</b>을 잡습니다.</p></div>
  </div>
  <ol class="plain">
    <li><b>미리 정한 규칙에서 1위였습니다.</b> 2025년 급락 90%를 잡을 때 헛경보 25일로, 이전 모델(Ridge+EBM, 31일)보다 19% 적어 교체 기준(15%)을 넘었습니다. 100%를 다 잡을 때도 122일에서 108일로 줄었습니다.</li>
    <li><b>랜덤포레스트는 단순하지만, 이 문제에 맞는 단순함입니다.</b> 학습할 급락일이 2024~2025년 27일뿐입니다. 나무를 순서대로 쌓아 오차를 끝까지 줄이는 부스팅(XGBoost·LightGBM)이나 딥러닝은 이 27일에 지나치게 맞춰져 새 연도에서 틀립니다. 랜덤포레스트는 서로 독립된 나무 500개를 평균하기 때문에 한 나무의 실수가 묻히고, 적은 자료에서도 잘 흔들리지 않습니다. 실제로 의사결정나무 하나는 최하위권(헛경보 214~346일)이었지만, 500개를 평균하자 단독으로도 58일이 됐습니다.</li>
    <li><b>서로 다른 방식이라 함께 쓸 때 효과가 컸습니다.</b> 랜덤포레스트 단독(58일)은 Ridge+EBM(31일)보다 못하지만, 직선·곡선·조건 조합이라는 서로 다른 시각을 섞자 한 모델이 놓친 날을 다른 모델이 잡아 25일이 됐습니다. 반대로 비슷한 방식끼리 섞은 부스팅 3종(63일)이나 신경망 4종(58일)은 섞어도 좋아지지 않았습니다.</li>
    <li><b>복잡한 모델은 효과가 없었습니다.</b> 딥러닝(LSTM 56일·Transformer 70일), 부스팅 단독(53~91일), TabPFN(49일) 모두 선형 모델(39~40일)보다도 못했습니다. Chronos는 과거 SMP만 보기 때문에 휴일·날씨·수요를 쓰지 못해 단독으로는 159~164일이었고, 앙상블에 넣으면 100% 잡기 기준은 좋아졌지만(67~88일) 90% 기준은 나빠졌습니다(43~77일).</li>
  </ol>

  <h3>2026년에도 더 좋았나</h3>
  <p>모델을 고른 뒤 2026년을 한 번 채점해, 이전 모델과 같은 기준으로 견줘 봤습니다. <b>2026년에는 두 모델의 성적이 거의 같았습니다.</b> 2025년에 보인 차이는 급락일이 적어 생긴 우연에 가까웠고, 랜덤포레스트를 더해 나빠진 점도 없습니다.</p>
  <div class="panel scroll"><table><thead><tr><th>2026년 (급락 20일)</th><th class="n">80% 잡을 때 헛경보</th><th class="n">90%</th><th class="n">95%</th><th class="n">100%</th><th class="n">ROC-AUC</th></tr></thead><tbody>
    <tr><td>이전: Ridge+EBM</td><td class="n">22</td><td class="n">31</td><td class="n">84</td><td class="n">88</td><td class="n">0.934</td></tr>
    <tr class="total"><td><b>최종: Ridge+EBM+랜덤포레스트</b></td><td class="n">21</td><td class="n">32</td><td class="n">86</td><td class="n">101</td><td class="n">0.931</td></tr>
  </tbody></table>
  <p class="muted" style="font-size:13.5px;margin-top:8px">문턱을 2026년 결과에 맞춰 옮겨 가며 본 값이라 운영 성적이 아니라 모델 능력 비교입니다. 100%에서 갈린 것은 급락일 하나(6/13 토요일, 최종 모델 예측 96.2원 / 이전 92.9원) 때문입니다.</p></div>
  <div class="panel scroll"><table><thead><tr><th>구간별 (90% 잡을 때 헛경보)</th><th class="n">급락</th><th class="n">이전</th><th class="n">최종</th><th class="n">ROC 이전 / 최종</th></tr></thead><tbody>
    <tr><td>2025 상반기</td><td class="n">9</td><td class="n">13</td><td class="n">11</td><td class="n">0.958 / 0.964</td></tr>
    <tr><td>2025 하반기</td><td class="n">10</td><td class="n">15</td><td class="n">14</td><td class="n">0.938 / 0.941</td></tr>
    <tr><td><b>2026 상반기</b></td><td class="n">15</td><td class="n">62</td><td class="n">64</td><td class="n">0.904 / 0.903</td></tr>
    <tr><td>2026 하반기</td><td class="n">5</td><td class="n">3</td><td class="n">2</td><td class="n">0.987 / 0.989</td></tr>
  </tbody></table>
  <p class="muted" style="font-size:13.5px;margin-top:8px">최종 모델이 4구간 중 3구간에서 1~2일 앞섰습니다. 2026 상반기는 두 모델 모두 똑같이 어려웠는데, 평소 SMP가 103원까지 내려가 평범한 날도 위험해 보였던 시기입니다. 날짜를 무작위로 다시 뽑아 1,000번 비교해도 차이는 우연 범위(헛경보 차이 95% 구간 −22~+20일)였고, 랜덤포레스트의 무작위성(시드 5개)에 따른 흔들림도 0~2일로 작았습니다.</p></div>
</section>
"""
    OUT.write_text(sec, encoding="utf-8")
    print("저장:", OUT, len(z), "개 후보")


if __name__ == "__main__":
    main()
