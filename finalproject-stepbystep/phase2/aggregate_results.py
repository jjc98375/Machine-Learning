import os
import glob
import json
import argparse
from phase2_config import OUTPUT_DIR

def format_score(val):
    if isinstance(val, (float, int)):
        return f"{val:.4f}"
    return str(val)

def aggregate(output_dir):
    json_files = glob.glob(os.path.join(output_dir, "*_results.json"))
    
    if not json_files:
        print(f"❌ Error: No *_results.json found in '{output_dir}'.")
        print("Please make sure you point to the folder containing your team's JSON files!")
        return

    print("=" * 100)
    print(" 🏆 TEAM MASTER COMPILATION: HYPERPARAMETER A/B TESTING RESULTS 🏆 ")
    print("=" * 100)
    print("\n| Run Configuration (Hyperparameters) | Backbone | In-Domain F1 | Zero-Shot F1 | Duration Acc |")
    print("|---|---|---|---|---|")
    
    # Sort files by name so epoch/samples align properly
    json_files.sort()
    
    for jf in json_files:
        # Prevent picking up the old generic JSON if it exists
        if os.path.basename(jf) == "performance_results.json": 
            continue
            
        with open(jf, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue
            
        run_id = data.get("run_id", os.path.basename(jf).replace("_results.json", ""))
        
        in_domain = data.get("in_domain", {})
        zero_shot = data.get("zero_shot", {})
        
        # Discover all models run in this JSON (e.g., xlm-roberta, mbert)
        backbones = set(list(in_domain.keys()) + list(zero_shot.keys()))
        
        for bb in sorted(list(backbones)):
            id_f1 = in_domain.get(bb, {}).get("mean_f1", "-")
            zs_f1 = zero_shot.get(bb, {}).get("mean_f1", "-")
            dur_acc = in_domain.get(bb, {}).get("duration_accuracy", "-")
            
            print(f"| `{run_id}` | **{bb}** | {format_score(id_f1)} | {format_score(zs_f1)} | {format_score(dur_acc)} |")
            
    print("\n✅ Copy and paste this table directly into your Final Report / Markdown file!\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregates JSON performance metrics from all team experiments.")
    parser.add_argument("--dir", type=str, default=OUTPUT_DIR, help="Directory containing the _results.json files to aggregate.")
    args = parser.parse_args()
    
    print(f"Scanning '{args.dir}' for results...\n")
    aggregate(args.dir)
