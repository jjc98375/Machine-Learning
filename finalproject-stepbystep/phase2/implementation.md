Context
Phase 1에서 데이터 파이프라인(스트리밍, 라벨링, subword 정렬, baseline)을 완성했고, 이제 Phase 2에서 실제 모델 학습 + 비교 실험 + 시각화를 구현해야 함. 발표 체크리스트 5개 항목을 모두 커버해야 함.
핵심 설계 결정

Phase 1 코드 재사용: sys.path로 phase1 import → CompleteStreamingDataset, collate_fn, compute_anticipatory_f1 그대로 활용
Causal masking: is_decoder=True (XLM-R, mBERT 둘 다 지원) → phase1의 수동 causal_mask.py 대신 사용
ignore_index: -100 (PyTorch 컨벤션, phase1과 일관)
데이터 수집: IterableDataset → list로 materialize → random_split으로 train/val 분리
MPS 지원: Apple Silicon Mac 대비 mps device fallback 포함

파일 구조
phase2/
├── config.py           # Phase 2 하이퍼파라미터
├── model.py            # Dual-head 모델 (XLM-R / mBERT 지원)
├── train.py            # 학습 루프 + 데이터 수집/분리
├── evaluate.py         # Per-pair F1, universality σ, duration accuracy
├── visualize.py        # Bar chart + convergence plots
└── run_experiment.py   # 메인 실행 (두 backbone 비교)
파일별 상세 계획
1. phase2/config.py

Phase 1 config import (PAIR_FILES, MAX_LENGTH, BATCH_SIZE 등)
학습 하이퍼파라미터: LR=2e-5, EPOCHS=5, WARMUP_RATIO=0.1
Loss 가중치: LAMBDA_SW=1.0, LAMBDA_DUR=0.5
두 backbone 정의: {"xlm-roberta": "xlm-roberta-base", "mbert": "bert-base-multilingual-cased"}
MAX_SAMPLES_PER_PAIR=2000 (6쌍 × 2000 = 12,000 총 샘플)
Output 경로 설정

2. phase2/model.py — PredictiveSwitchModel(nn.Module)

__init__(model_name, lambda_sw, lambda_dur):

AutoConfig.from_pretrained(model_name) + config.is_decoder = True
AutoModel.from_pretrained(model_name, config=config) → 두 backbone 통합 지원
switch_head: Linear(hidden, 1) + BCEWithLogitsLoss(reduction='none')
duration_head: Linear(hidden, 3) + CrossEntropyLoss(ignore_index=-100)


forward(input_ids, attention_mask, switch_labels=None, duration_labels=None):

개별 loss 반환: loss, loss_sw, loss_dur (convergence plot용)
Switch: active_mask로 -100 필터 후 BCE → mean
Duration: CE(ignore_index=-100)이 자동 처리
L_total = λ_sw * L_sw + λ_dur * L_dur



3. phase2/train.py

collect_dataset(model_name, max_per_pair): phase1 스트리밍 → list로 수집 + 디스크 캐시(.pt)
ListDataset(Dataset): list를 map-style Dataset으로 래핑
train_model(model_name) -> dict:

데이터 수집 → ListDataset → random_split (80/20)
DataLoader with phase1의 collate_fn
AdamW + linear warmup scheduler
Gradient clipping (max_norm=1.0)
epoch별 train/val loss 기록 (total, sw, dur 각각)
체크포인트 저장
반환: {"model", "history", "val_loader"}



4. phase2/evaluate.py

evaluate_model(model, dataloader, device) -> dict:

Switch: sigmoid > 0.5 → per-pair F1 (phase1의 compute_anticipatory_f1 재사용)
Duration: argmax → accuracy + macro F1 (switch 발생 위치만)
반환: per_pair_f1, sigma, mean_f1, f1_switch, duration_accuracy


compare_models(results_xlmr, results_mbert): 메트릭 비교 테이블
print_evaluation_report(results, model_name): 포맷된 출력

5. phase2/visualize.py

plot_f1_bar_chart(results_dict, save_path): 6개 언어쌍 × 2 backbone 그룹 바차트 + σ 표시
plot_convergence(history, model_name, save_path): 2개 서브플롯 (switch loss, duration loss)
plot_comparison_convergence(h_xlmr, h_mbert, save_path): 총 loss 오버레이

6. phase2/run_experiment.py

main(): XLM-R 학습→평가 → mBERT 학습→평가 → 비교 → 시각화
CLI args로 epochs, samples, backbone 선택 가능

의존 관계 (Phase 1 → Phase 2)
Phase 2 파일Phase 1에서 importconfig.pyconfig.PAIR_FILES, MAX_LENGTH, BATCH_SIZEtrain.pydataset.CompleteStreamingDataset, dataset.collate_fnevaluate.pybaseline.compute_anticipatory_f1model.py없음 (순수 PyTorch + transformers)
발표 체크리스트 매핑
항목점수커버Dual-Head Architecture2점model.py: switch_head + duration_head on shared backboneLoss Function Optimization3점model.py: λ 가중치 + ignore_index=-100Experiment A: Comparative Study2점run_experiment.py: XLM-R vs mBERT 비교Initial Universality Audit2점visualize.py: per-pair F1 bar chart + σLMS Preparation1점visualize.py: convergence plots (switch/duration 각각)
검증 방법

먼저 python run_experiment.py --epochs 1 --samples_per_pair 100으로 빠른 smoke test
전체 실행: python run_experiment.py (5 epochs, 2000 samples/pair)
outputs/plots/ 폴더에 bar chart + convergence plot 생성 확인
val loss가 epoch별로 감소하는지 확인 (convergence)
per-pair F1이 baseline(naive=always 0) 대비 향상되었는지 확인