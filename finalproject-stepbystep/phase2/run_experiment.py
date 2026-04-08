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
    parser.add_argument("--unfreeze_layers", type=int, default=3, help="Number of top backbone layers to unfreeze (0=full fine-tune)")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--resume_path", type=str, default=None, help="Path to saved model checkpoint (.pt)")
    parser.add_argument("--single_task", action="store_true", help="Switch-only ablation (disable duration head loss)")
    return parser.parse_args()

def main():
    args = parse_args()
    device = get_device()
    
    results = {}
    zero_shot_results = {}
    histories = {}
    
    # Build unique run name from all params + zero-shot config
    zs_tag = ""
    if args.zero_shot_pairs:
        zs_short = "_".join(p.split("-")[0][:3] for p in args.zero_shot_pairs)  # e.g. "Fre_Spa"
        zs_tag = f"_zs-{zs_short}"
    else:
        zs_tag = "_supervised"
    
    ul_tag = f"_ul{args.unfreeze_layers}" if args.unfreeze_layers > 0 else "_fullft"
    st_tag = "_singletask" if args.single_task else ""
    current_run_dir_name = f"run_ep{args.epochs}_s{args.samples_per_pair}_bs{args.batch_size}_lr{args.lr}_a{args.focal_alpha}_g{args.focal_gamma}{ul_tag}{zs_tag}{st_tag}"
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
            unfreeze_layers=args.unfreeze_layers,
            patience=args.patience,
            run_name=current_run_dir_name,
            single_task=args.single_task
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
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n[{now}] === EXPERIMENT RUN SUMMARY ===\n")
        f.write(f"Hyperparameters:\n")
        f.write(f" - Epochs                   : {args.epochs}\n")
        f.write(f" - Max Samples per Pair     : {args.samples_per_pair}\n")
        f.write(f" - Batch Size               : {args.batch_size}\n")
        f.write(f" - Learning Rate            : {args.lr}\n")
        f.write(f" - Focal Loss (Alpha)       : {args.focal_alpha}\n")
        f.write(f" - Focal Loss (Gamma)       : {args.focal_gamma}\n")
        f.write(f" - Unfreeze Layers          : {args.unfreeze_layers}\n")
        f.write(f" - Early Stop Patience      : {args.patience}\n")
        if args.zero_shot_pairs:
            f.write(f" - Zero-Shot Held Out       : {args.zero_shot_pairs}\n")
        f.write(f"\n")
        
        f.write(f"Results Breakdown:\n")
        for model_name_key, res in results.items():
            mac_f1 = res.get('mean_f1', 0)
            dur_acc = res.get('duration_accuracy', 0)
            sigma = res.get('sigma', 0)
            sw_f1 = res.get('f1_switch', 0)
            nosw_f1 = res.get('f1_no_switch', 0)
            
            f.write(f" [{model_name_key.upper()}]\n")
            f.write(f"   * Mean Anticipatory F1 : {mac_f1:.4f}\n")
            f.write(f"   * Universality (Sigma) : {sigma:.4f}\n")
            f.write(f"   * Duration Accuracy    : {dur_acc:.4f}\n")
            f.write(f"   * F1 (Switch=1)        : {sw_f1:.4f}\n")
            f.write(f"   * F1 (No-Switch=0)     : {nosw_f1:.4f}\n")
            
            # 언어 쌍별 자세한 점수도 교수님 보시기 좋게 영어로 정리해줍니다.
            if 'per_pair_f1' in res:
                f.write(f"     [Per-Pair F1 Breakdown]\n")
                for p_name, p_f1 in res['per_pair_f1'].items():
                    f.write(f"      - {p_name:<16}: {p_f1:.4f}\n")
            
            # Zero-Shot 결과 로깅
            if args.zero_shot_pairs and model_name_key in zero_shot_results:
                zs_res = zero_shot_results[model_name_key]
                f.write(f"\n     [ZERO-SHOT UNIVERSAL EVALUATION]\n")
                f.write(f"      * Mean ZS F1   : {zs_res.get('mean_f1', 0):.4f}\n")
                for p_name, p_f1 in zs_res.get('per_pair_f1', {}).items():
                    f.write(f"      - {p_name:<16}: {p_f1:.4f} (Unseen Data!)\n")
            
            f.write("\n")
        f.write("=" * 65 + "\n")
    print(f"\n✅ 현재 실험 결과 영문 프레젠테이션용 명세서가 'experiment_history.txt' 에 기록되었습니다!")
    
    # JSON 파일로도 내보내기 (이름을 고유하게 저장!)
    json_path = os.path.join(OUTPUT_DIR, f"{current_run_dir_name}_results.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump({
            "run_id": current_run_dir_name,
            "hyperparameters": {
                "epochs": args.epochs,
                "samples": args.samples_per_pair,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "focal_alpha": args.focal_alpha,
                "focal_gamma": args.focal_gamma,
                "unfreeze_layers": args.unfreeze_layers,
                "patience": args.patience,
                "zero_shot_pairs": args.zero_shot_pairs
            },
            "in_domain": results,
            "zero_shot": zero_shot_results
        }, jf, indent=4)
    print(f"✅ 성능 데이터 고유 JSON 덤프 완료: '{current_run_dir_name}_results.json'")

if __name__ == "__main__":
    main()
