import argparse
import os
import torch
import datetime 
import json

from phase2_config import MODELS, PLOTS_DIR, OUTPUT_DIR, EPOCHS, MAX_SAMPLES_PER_PAIR, BATCH_SIZE, LR
from train import train_model, get_device
from evaluate import evaluate_model, print_evaluation_report, compare_models
from visualize import plot_f1_bar_chart, plot_convergence, plot_comparison_convergence

def parse_args():
    parser = argparse.ArgumentParser(description="Run SwitchLingua Phase 2 Experiment")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--samples_per_pair", type=int, default=MAX_SAMPLES_PER_PAIR, help="Max samples per language pair")
    parser.add_argument("--backbones", nargs="+", choices=["xlm-roberta", "mbert"], 
                        default=["xlm-roberta", "mbert"], help="Backbones to run")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Training and validation batch size")
    parser.add_argument("--lr", type=float, default=LR, help="AdamW learning rate")
    parser.add_argument("--focal_alpha", type=float, default=0.8, help="Focal Loss Alpha (weight for positive class)")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal Loss Gamma (focusing parameter)")
    parser.add_argument("--zero_shot_pairs", nargs="*", default=[], help="Language pairs to hold out for Zero-Shot Evaluation")
    parser.add_argument("--resume_path", type=str, default=None, help="과거 저장된 뇌 파일(.pt)의 상대 또는 절대 경로")
    return parser.parse_args()

def main():
    args = parse_args()
    device = get_device()
    
    results = {}
    zero_shot_results = {}
    histories = {}
    
    # ✨ 1. 회원님 요청: 실험 조건(하이퍼파라미터)에 따라 자동으로 구분되는 전용 결과 폴더 생성!
    current_run_dir_name = f"run_ep{args.epochs}_s{args.samples_per_pair}_bs{args.batch_size}_lr{args.lr}_a{args.focal_alpha}_g{args.focal_gamma}"
    custom_plots_dir = os.path.join(PLOTS_DIR, current_run_dir_name)
    os.makedirs(custom_plots_dir, exist_ok=True)
    
    for short_name in args.backbones:
        model_name = MODELS[short_name]
        print(f"\n{'='*80}")
        print(f"RUNNING EXPERIMENT: {short_name} ({model_name})")
        print(f"{'='*80}")
        
        # 1. Train (Excluding Zero-Shot Pairs and using new hyperparameters)
        train_res = train_model(
            model_name, 
            epochs=args.epochs, 
            max_samples_per_pair=args.samples_per_pair, 
            resume_path=args.resume_path,
            exclude_pairs=args.zero_shot_pairs,
            batch_size=args.batch_size,
            lr=args.lr,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            run_name=current_run_dir_name
        )
        model = train_res["model"]
        history = train_res["history"]
        val_loader = train_res["val_loader"]
        
        # 2. Evaluate In-Domain (Seen Languages)
        print("\n[Evaluating In-Domain Validation Set]")
        eval_res = evaluate_model(model, val_loader, device)
        print_evaluation_report(eval_res, short_name + " (In-Domain)")
        
        # 2.5 Evaluate Zero-Shot (Unseen Languages)
        if args.zero_shot_pairs:
            from train import get_eval_dataloader
            # Collect 500 samples per zero-shot pair for quick but robust evaluation
            zs_loader = get_eval_dataloader(model_name, min(500, args.samples_per_pair), include_pairs=args.zero_shot_pairs)
            zs_res = evaluate_model(model, zs_loader, device)
            print_evaluation_report(zs_res, short_name + " (ZERO-SHOT)")
            zero_shot_results[short_name] = zs_res
        
        # 3. Visualize individual (방금 팠던 전용 새 폴더에 쏙 저장!)
        plot_convergence(history, short_name, os.path.join(custom_plots_dir, f"{short_name}_convergence.png"))
        
        results[short_name] = eval_res
        histories[short_name] = history
        
    # 4. Compare if both models were run (이것도 전용 폴더에 쏙!)
    if len(args.backbones) == 2 and "xlm-roberta" in results and "mbert" in results:
        compare_models(results)
        plot_f1_bar_chart(results, os.path.join(custom_plots_dir, "f1_comparison.png"))
        plot_comparison_convergence(
            histories["xlm-roberta"], histories["mbert"],
            os.path.join(custom_plots_dir, "total_loss_comparison.png")
        )

    # =========================================================================
    # ✨ 5. [회원님 요청 기능 2] 리포트용 엄청나게 자세한 영문판 텍스트 요약지!
    # =========================================================================
    log_path = os.path.join(OUTPUT_DIR, "experiment_history.txt")
    with open(log_path, "a", encoding="utf-8") as f:
        now =