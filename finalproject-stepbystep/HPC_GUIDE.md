# NEU Explorer HPC Guide for Group D Experiments

**Cluster:** NEU Explorer HPC
**Login:** `ssh cho.jae@login.explorer.northeastern.edu`
**Scheduler:** SLURM

---

## Prerequisites (One-Time Setup)

### 1. SSH into the HPC
```bash
ssh cho.jae@login.explorer.northeastern.edu
```

### 2. Clone the repo (if not already done)
```bash
cd ~
git clone https://github.com/jjc98375/Machine-Learning.git
```

### 3. Create the conda environment (if `MLenv` doesn't exist yet)
```bash
module load anaconda3/2024.06
conda create -n MLenv python=3.10 -y
conda activate MLenv
pip install torch transformers datasets langid scikit-learn tqdm matplotlib huggingface_hub
```

### 4. HuggingFace authentication (required for gated dataset)
```bash
conda activate MLenv
huggingface-cli login
# Paste your HF access token when prompted
# Token must have Read access to Shelton1013/SwitchLingua_text
```

### 5. Verify the environment works
```bash
module load anaconda3/2024.06
conda activate MLenv
python -c "import torch; print(torch.cuda.is_available())"
python -c "from datasets import load_dataset; print('datasets OK')"
python -c "from transformers import AutoModel; print('transformers OK')"
```

---

## Running Group D Experiments

### Pull latest code
```bash
cd ~/Machine-Learning
git pull origin main
```

### Submit all 5 jobs
```bash
cd ~/Machine-Learning/finalproject-stepbystep
sbatch slurm/run_d1.sh   # Zero-shot Latin (hold out French, Spanish)
sbatch slurm/run_d2.sh   # Zero-shot CJK (hold out Chinese, Japanese)
sbatch slurm/run_d3.sh   # Zero-shot Cyrillic+Arabic (hold out Russian, Arabic)
sbatch slurm/run_d4.sh   # Zero-shot Hindi+Korean (hold out Hindi, Korean)
sbatch slurm/run_d5.sh   # Full supervised (all 10 pairs, 10K samples)
```

### Monitor jobs
```bash
squeue -u cho.jae                    # List all your running/pending jobs
scancel <job_id>                     # Cancel a specific job
scancel -u cho.jae                   # Cancel ALL your jobs
tail -f slurm/d1_latin_<jobid>.out   # Live output of D1
```

### Check results after completion
```bash
# Experiment history (all metrics logged here)
cat ~/Machine-Learning/finalproject-stepbystep/phase2/outputs/experiment_history.txt

# Plots are saved in:
ls ~/Machine-Learning/finalproject-stepbystep/phase2/outputs/plots/

# JSON results (if aggregate_results.py was run):
ls ~/Machine-Learning/finalproject-stepbystep/phase2/outputs/*.json
```

### Download results to local machine (run from YOUR laptop, not HPC)
```bash
scp cho.jae@login.explorer.northeastern.edu:~/Machine-Learning/finalproject-stepbystep/phase2/outputs/experiment_history.txt ~/Downloads/
scp -r cho.jae@login.explorer.northeastern.edu:~/Machine-Learning/finalproject-stepbystep/phase2/outputs/plots/ ~/Downloads/hpc_plots/
```

---

## SLURM Script Details

All scripts are in `slurm/` and share these settings:

| Setting | Value |
|---------|-------|
| Partition | gpu |
| GPUs | 1 |
| Memory | 32G |
| Module | anaconda3/2024.06 |
| Conda env | MLenv |
| Model | mBERT only |
| Focal Loss | alpha=0.8, gamma=2.0 |

| Script | Job Name | Zero-Shot Pairs | Samples/Pair | Epochs | Time Limit |
|--------|----------|----------------|-------------|--------|------------|
| run_d1.sh | groupD1_latin | French-English, Spanish-English | 8000 | 5 | 4 hrs |
| run_d2.sh | groupD2_cjk | Chinese-English, Japanese-English | 8000 | 5 | 4 hrs |
| run_d3.sh | groupD3_cyrillic_arabic | Russian-English, Arabic-English | 8000 | 5 | 4 hrs |
| run_d4.sh | groupD4_hindi_korean | Hindi-English, Korean-English | 8000 | 5 | 4 hrs |
| run_d5.sh | groupD5_full_supervised | None (all 10 pairs) | 10000 | 5 | 6 hrs |

---

## Troubleshooting

### Job stuck in PENDING
```bash
squeue -u cho.jae   # Check reason column
# Common reasons:
# Resources - waiting for GPU, just wait
# QOSMaxJobsPerUser - too many jobs, cancel one or wait
```

### Out of memory (OOM)
Edit the SLURM script: change `--mem=32G` to `--mem=48G` or reduce `--samples_per_pair`.

### Module not found
```bash
module avail anaconda   # Check available anaconda versions
module avail cuda       # May need to load CUDA explicitly
```

### conda activate fails in SLURM
Add this before `conda activate` in the script:
```bash
eval "$(conda shell.bash hook)"
```

### HuggingFace authentication fails in job
The HF token may not persist in batch jobs. Add this to the SLURM script before the python command:
```bash
export HF_TOKEN="your_token_here"
```
Or create `~/.huggingface/token` file with your token.

### CUDA not detected
```bash
# Check if GPU is allocated
nvidia-smi
# If empty, the job may not have gotten a GPU - check SLURM output for errors
```

---

## After All Jobs Complete

1. Download `experiment_history.txt` - contains all Group D metrics
2. Download plots from `outputs/plots/`
3. Feed results back to the main project for `PRESENTATION_DRAFT.md` Slides 11-12 (zero-shot results)
4. Compare zero-shot F1 vs supervised F1 from Groups A/C (see `EXPERIMENT_ANALYSIS.md`)
