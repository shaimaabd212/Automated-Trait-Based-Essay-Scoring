# Automated Trait-Based Essay Scoring

## 1. About the project

You paste an English essay and the app gives it a score for 6 traits: **overall, content, organization,
word choice, sentence fluency and conventions**. For each score it also shows how confident the model is.

We follow the method of one paper: Xiao et al. (2025), *Human-AI Collaborative Essay Scoring: A Dual-Process
Framework with LLMs*.

The project uses one model, **LLaMA-3-8B**, fine-tuned with LoRA, but it scores an essay in two ways:

| | Fast module | Slow module |
|---|---|---|
| **How it scores** | A small classifier reads the model's numbers (embedding) for the essay | The model writes the scores as text |
| **Speed** | Under 1 second | A few seconds |
| **When it is used** | Always, first | Only for traits where the fast module is not sure (confidence below 0.2) |

**Data:** 1,783 student essays from the ASAP dataset (set 1, topic: effects of computers on people). ASAP gives
the overall score; ASAP++ adds the trait scores.

> ⚠️ The model was trained on ASAP set 1 only. Scores for essays on other topics are not reliable.

### Results

Agreement with the teachers' scores, measured with QWK (1 = same as the teachers, 0 = random), on the 20% test set:

| Trait | Fast module | Dual process (fast + slow) |
|---|---|---|
| Overall | 0.786 | 0.786 |
| Content | 0.730 | 0.728 |
| Organization | 0.708 | 0.708 |
| Word choice | 0.647 | 0.647 |
| Sentence fluency | 0.669 | 0.669 |
| Conventions | 0.667 | 0.667 |
| **Average** | **0.701** | **0.701** |

For comparison, the paper reports 0.758 (fast) and 0.761 (dual process) for the overall score.

## 2. File structure

```
Automated-Trait-Based-Essay-Scoring/
├── app/
│   ├── app.py                  # The Streamlit web app (paste an essay, see the scores)
│   └── scorer.py               # Loads the trained model and scores one essay (used by app.py)
├── requirements.txt            # Python libraries to install
├── .gitignore                  # Files git should not push (e.g. .venv/)
├── data/
│   ├── training_set_rel3.tsv   # ASAP essays and overall scores
│   └── Prompt-1.csv … Prompt-6.csv   # ASAP++ trait scores
├── models/                     # Trained model (large files are stored with Git LFS)
│   ├── llama3_aes_lora/        # LoRA weights + tokenizer for LLaMA-3-8B (slow module)
│   ├── fast_heads.joblib       # Fast-module classifiers + the prompt text used in training
│   └── embeddings.npz          # Essay embeddings saved during training
├── notebooks/
│   ├── dual_process_aes.ipynb  # Trains and tests the model (Colab, T4 GPU, ~1.5 hours)
│   └── Essay_Scoring.ipynb     # Runs the Streamlit app on Colab
├── results/
│   └── results_qwk.csv         # QWK scores in the table above
└── references/                 # The papers we read
```

## 3. How to run the app on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaimaabd212/Automated-Trait-Based-Essay-Scoring/blob/main/notebooks/Essay_Scoring.ipynb)

The model needs a GPU with more than 6 GB of memory, so we run it on Colab's free T4 GPU. The notebook
(`notebooks/Essay_Scoring.ipynb`) reads the app and the model from your Google Drive, so you need to put them
there first.

### Step 1: Put the files in your Google Drive (one time only)

1. **Get the files from this repo.** The model files are large and stored with Git LFS, so **do not use
   "Code → Download ZIP"** (it only gives small placeholder files). Use one of these:
   - **With git:** install [Git LFS](https://git-lfs.com), then run
     `git clone https://github.com/shaimaabd212/Automated-Trait-Based-Essay-Scoring.git`
   - **Without git:** on GitHub, open each file you need and click the **Download raw file** button (↓).
2. In [Google Drive](https://drive.google.com), inside **My Drive**, create a folder named exactly
   **`essay_scoring`**.
3. Upload these into the `essay_scoring` folder:
   - `app.py` and `scorer.py` from the repo's `app/` folder. Put them **directly** in `essay_scoring`, not in an
     `app` subfolder.
   - The whole `models` folder.

   Your Drive should look like this:

   ```
   My Drive/
   └── essay_scoring/
       ├── app.py
       ├── scorer.py
       └── models/
           ├── fast_heads.joblib
           └── llama3_aes_lora/
               ├── adapter_config.json
               ├── adapter_model.safetensors
               ├── tokenizer.json
               └── tokenizer_config.json
   ```

4. Wait until the upload is finished. In Drive, check that `adapter_model.safetensors` is about **160 MB**. If it
   is smaller, the upload is not done yet and the app will fail.

### Step 2: Run the notebook

1. Click the **Open in Colab** button above.
2. Go to **Runtime → Change runtime type**, check that **T4 GPU** is selected, and click **Save**.
3. Click **Runtime → Run all**. When Colab asks for access to your Google Drive, allow it.
4. The 5th cell starts the app and keeps running. In its output, find the link that ends with
   **`.trycloudflare.com`** and open it.
5. Paste an essay (at least 50 words) and click **Score essay**. The first essay is slow (a few minutes) because
   the base model (about 5 GB) is being downloaded and loaded. After that, each essay takes a few seconds.

**Good to know**
- Keep the Colab tab open. The link stops working when the Colab session ends, and you get a new link each time.
- The 5th cell keeps running while the app is on. That is normal; stop it to shut the app down.
- If the page shows an error, run `!tail -n 20 /content/drive/MyDrive/essay_scoring/log.txt` in a new cell to
  see what went wrong.
- The cells after the 5th one are only for fixing problems. If Drive disconnects or the model file in Drive is
  incomplete, run them: they reconnect Drive, copy the folder to Colab's own disk, and restart the app from there.
- Changes you make in the notebook are not saved to GitHub. To keep them, use **File → Save a copy in Drive**.

## 4. How we trained the model

Run `notebooks/dual_process_aes.ipynb` in Google Colab with a T4 GPU. It takes about 1.5 hours.

1. **Set up (Steps 1–2):** install the libraries, connect Google Drive, set the file paths.
2. **Load the data (Step 3):** join ASAP and ASAP++ so each essay has all its scores.
3. **Split (Step 4):** 80% of essays to train, 20% to test.
4. **Make the prompt (Step 5):** each essay becomes one text: instructions, rubric, essay, and the correct scores.
5. **Train LLaMA-3-8B (Steps 6–7):** it learns to write the right scores. This is the slow module.
6. **Test the slow module (Step 8):** the model writes scores for an essay.
7. **Turn essays into numbers (Step 9):** the model changes each essay into 4,096 numbers (an embedding).
8. **Train the fast module (Step 10):** one small classifier per trait learns scores from those numbers.
9. **Confidence (Step 11):** for each score, 0 = just guessing, 1 = completely sure.
10. **Combine (Step 12):** keep the fast score; if confidence is below 0.2, ask the slow module.
11. **Measure (Steps 13–14):** compare our scores with the teachers' using QWK.
12. **Save (Step 15):** save the trained model and classifiers for the app.

## 5. How the app scores an essay

1. The user pastes an essay and clicks **Score essay**.
2. The app checks that the essay has at least 50 words.
3. The fast module scores every trait and gives a confidence for each one.
4. For traits with confidence below 0.2, the slow module writes new scores.
5. The page shows the overall score and a table: score, confidence, and which module (fast or slow) gave it.


  `git rm -r --cached .venv`, then commit.
- Large model files (`*.safetensors`, `*.npz`) are sent with Git LFS automatically. Install
  [Git LFS](https://git-lfs.com) before you clone or push. GitHub does not accept files over 100 MB without it.
- Pull before you start working (`git pull`) so you have your teammates' latest changes.
