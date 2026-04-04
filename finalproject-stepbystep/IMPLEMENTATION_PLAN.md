# SwitchLingua Final Project: Targeted Implementation Plan
**Updated for Final Defense: Addressing Direct Professor Feedback**

---

## 1. Addressing Professor Feedback

### Feedback 1: "Increase more language pairs"
**Current state**: 6 language pairs.
**Action Plan**: We will update `phase1/config.py` to securely add 4-6 new language pairs (e.g., German, Italian, Russian, Turkish, Japanese, Vietnamese) to reach 10-12 total pairs. This proves the robustness of the data pipeline.

### Feedback 2: "Instead of having each language with each model, have one model that generally predicts the switch"
**Current state**: Our system *already* trains a single universal mBERT/XLM-R model! However, to **PROVE** to the professor that the model "generally predicts the switch" rather than just memorizing vocabulary from the training set, we need a rigorous **Zero-Shot Universal Evaluation**.
**Action Plan**:
- Introduce a `--zero_shot_pairs` argument in `run_experiment.py`.
- If 10 pairs exist, we train the model strictly on 8 pairs, hiding the remaining 2 completely.
- During evaluation, we test on ALL 10 pairs.
- If the model successfully predicts code-switching on the 2 *unseen* languages, this scientifically proves it is a Universal Predictor!

### Feedback 3: "Sum should be equal to 1. Loss weight always < 1.0"
**Current state**: We recently patched this to `lambda_sw=0.67` and `lambda_dur=0.33`.
**Action Plan**: Ensure that our code, our evaluation scripts, and the final presentation strictly mention that our loss functions represent a purely normalized convex combination ($\sum \lambda_i = 1.0$).

---

## 2. Technical Execution Roadmap

### Task 1: Expand Dataset & Prove Zero-Shot Universality
1. **Modify `phase1/config.py`**: Add new pairs to `PAIR_FILES`.
2. **Modify `phase2/train.py`**: Update the `collect_dataset` function to accept an `exclude_pairs` list. The dataloader must strictly skip these held-out linguistic pairs during training.
3. **Modify `phase2/run_experiment.py`**: Introduce `--zero_shot_pairs` so we can easily command the script to split the dataset dynamically.
4. **Modify `phase2/visualize.py`**: Highlight the Zero-Shot test sets in the generated bar charts with a distinct color (e.g., gold vs blue) to visually demonstrate "Universal Prediction" on unseen languages.

### Task 2: Re-Run the Final Full-Scale Training
Run the ultimate command on Colab's T4 GPU:
```bash
python run_experiment.py --epochs 5 --samples_per_pair 2000 --zero_shot_pairs French-English Spanish-English
```
This isolates the Same-Script languages to see if the model can magically deduce them just by learning from Distant Typologies (like Chinese-English).

### Task 3: Develop Qualitative Analysis for Presentation (Task 3 from Old Plan)
To show the professor what the model *actually* does:
- Implement `qualitative_analysis.py` to extract highly readable examples:
  - **True Positives**: Output sentences where the model perfectly predicted a switch.
  - **False Positives**: Output sentences where the model got confused (great for "Limitations" slide).

### Task 4: Interactive Live Demo (Task 4 from Old Plan)
- Create `demo.py` to load the trained `<model_final.pt>`.
- Allow live terminal inputs where the model prints its `Switch Probability` token by token as someone types.

---

## 3. Timeline to April 22 Defense
- **Step 1 (Now)**: Code the Zero-Shot architecture and expand `config.py`.
- **Step 2**: Run the overnight Colab Training (saving outputs to Drive).
- **Step 3**: Use the generated F1 plots and Qualitative examples to populate the Final Presentation Slides.
