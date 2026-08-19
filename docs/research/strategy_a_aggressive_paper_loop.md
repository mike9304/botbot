# Strategy A 공격형 페이퍼 루프 (연구 전용)

**상태:** 스크립트 준비됨. 공개 OHLCV 백테스트 숫자는 `research/strategy_a_aggressive/results/study.json` 생성 후 이 문서를 갱신한다.  
**금지:** 거래소 API 키, 실주문, Bitget 로그인. Grok 봇은 시그널/워치만.

오늘(UTC) 기준 윈도우: last-12m = 2025-08-19~2026-08-19, 2022-now = 2022-01-01~, full = 2020-01-01~.

---

## 1. 레포에서 재사용한 것

| 모듈 | 재사용 | 이번 연구에서 한 일 |
|---|---|---|
| `src/data/market_data.py` | Binance 공개 kline 패턴 | Bitget USDT-FUTURES 공개 캔들 + Binance Vision 폴백. 키 없음. |
| `src/strategies/momentum.py` | ATR 개념 | Wilder ATR (TradingView RMA). 레포 SMA ATR은 공식 A v1.1과 다름. |
| `src/strategies/advanced.py` | ADX 국면 | Wilder ADX. 레포는 SMA DX. |
| `src/core/exchange.py` | 수수료/격리 청산 개념 | 공식 페이퍼 스펙 0.20% RT. 바이낸스 0.05% taker와 다름. |
| `src/core/simulation.py` | 멀티에이전트 시뮬 | **사용 안 함.** 한 포지션·리스크% 사이징이 필요해서 별도 엔진. |

프로덕션 트레이딩 봇을 추가하지 않았다.

---

## 2. 공식 Strategy A v1.1 (검증 베이스라인)

- 4H Donchian 55 돌파 + 1D EMA50/200 국면 + 4H ADX(14)>25
- 손절 2×ATR(14,4H), 트레일 Donchian(10) 반대 밴드 (래칫)
- 리스크 0.5%/거래, 격리, 유효 레버리지 ≤3x, 동시 1포지션
- 시그널=봉 종가, 체결=다음 시가, 왕복 비용 0.20%

사용자 제시 공식 숫자 (이 런에서 **재계산하지 않음**, 참조만):

| 구간 | Return | WR | 기타 |
|---|---:|---:|---|
| last-12m | +2.17% | 25% (12 trades) | |
| 2022-now | +11.36% | — | |
| full 2020-2026 | +27.27% | 40% | MDD 4.8% |

이번 엔진이 위 숫자를 그대로 재현하지 않을 수 있다 (ADX 스무딩, Donchian 현재봉 제외, 일봉 정렬, Bitget vs 다른 벤더). **지어낸 WR/수익은 쓰지 않는다.**

---

## 3. 공격형 변형 5개 (리스크 캡 유지)

공통: 격리, 유효 레버 ≤3x, 1포지션, next-open, 0.20% RT, 1D EMA50/200 국면.

| ID | 아이디어 | TF | Donchian | ADX | R | ATR stop | 추가 |
|---|---|---|---:|---|---:|---:|---|
| V1 | 채널 단축 + 약한 ADX + 리스크↑ | 4H | 20 | >20 | 1.0% | 1.5 | — |
| V2 | ADX 제거 (빈도↑) | 4H | 20 | off | 1.0% | 1.5 | — |
| V3 | 더 짧은 터틀 | 4H | 10 | >20 | 1.0% | 1.5 | — |
| V4 | 1H 돌파 (데이터 있으면) | 1H | 20 | >20 | 1.0% | 1.5 | — |
| V5 | 2H + 모멘텀 오버레이 | 2H | 20 | >20 | 1.0% | 1.5 | ROC(12) 방향 + close vs EMA20 |

---

## 4. 백테스트 숫자

**아직 없음.** `python -m research.strategy_a_aggressive.run_study` 실행 후 이 절을 실제 숫자로 교체한다. 데이터를 못 받으면 그 사실을 명시한다.

추천 페이퍼 후보 / last-12m vs 2022-now vs full / MDD / 과적합 경고 — 실측 후 작성.

---

## 5. 실행

```bash
python -m research.strategy_a_aggressive.test_engine
python -m research.strategy_a_aggressive.run_study
```

결과: `research/strategy_a_aggressive/results/study.json`
