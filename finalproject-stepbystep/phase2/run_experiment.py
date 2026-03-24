import argparse
import os
import torch
import datetime 

from phase2_config import MODELS, PLOTS_DIR, OUTPUT_DIR, EPOCHS, MAX_SAMPLES_PER_PAIR
from train import train_model, get_device
from evaluate import evaluate_model, print_evaluation_report, compare_models
from visualize import plot_f1_bar_chart, plot_convergence, plot_comparison_convergence

def parse_args():
    parser = argparse.ArgumentParser(description="Run SwitchLingua Phase 2 Experiment")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--samples_per_pair", type=int, default=MAX_SAMPLES_PER_PAIR, help="Max samples per language pair")
    parser.add_argument("--backbones", nargs="+", choices=["xlm-roberta", "mbert"], 
                        default=["xlm-roberta", "mbert"], help="Backbones to run")
    return parser.parse_args()

def main():
    args = parse_args()
    device = get_device()
    
    results = {}
    histories = {}
    
    # ✨ 1. 회원님 요청: 에포크와 샘플 크기에 따라 구분되는 새 폴더 방을 알아서 만듭니다!
    current_run_dir_name = f"run_ep{args.epochs}_s{args.samples_per_pair}"
    custom_plots_dir = os.path.join(PLOTS_DIR, current_run_dir_name)
    os.makedirs(custom_plots_dir, exist_ok=True)
    
    for short_name in args.backbones:
        model_name = MODELS[short_name]
        print(f"\n{'='*80}")
        print(f"RUNNING EXPERIMENT: {short_name} ({model_name})")
        print(f"{'='*80}")
        
        # 1. Train
        train_res = train_model(model_name, epochs=args.epochs, max_samples_per_pair=args.samples_per_pair)
        model = train_res["model"]
        history = train_res["history"]
        val_loader = train_res["val_loader"]
        
        # 2. Evaluate
        eval_res = evaluate_model(model, val_loader, device)
        print_evaluation_report(eval_res, short_name)
        
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
        f.write(f" - Approximate Total Samples: {args.samples_per_pair * 6}\n\n")
        
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
            f.write("\n")
        f.write("=" * 65 + "\n")
    print("\n✅ 현재 실험 결과 영문 프레젠테이션용 명세서가 'experiment_history.txt' 에 기록되었습니다!")

if __name__ == "__main__":
    main()
