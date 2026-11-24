# Automated-Trait-Based-Essay-Scoring

# 1- About the project

This project allows the student to paste an English essay and get the score for 6 traits:  overall,
content, organization, word choice, sentence fluency and conventions.

We copy the method of one paper: Xiao et al. (2025), *Human-AI Collaborative Essay Scoring: A Dual-Process Framework with LLMs*.

The project uses one model which is LLaMa-3-8B, but uses two ways to score the essay.

**Data:** 1,783 student essays from the ASAP dataset (set 1, topic: effects of computers on people). ASAP gives the overall score; ASAP++ adds the trait scores.

## 2. How we train it

Run the notebook `dual_process_aes.ipynb` in Google Colab with a T4 GPU. It takes about 1.5 hours.

1. **Set up (Steps 1–2):** install the library, connect Google Drive, set the file paths.
2. **Load the data (Step 3):** join ASAP and ASAP++ so each essay has all its scores.
3. **Split (Step 4):** 80% of essays to train, 20% to test.
4. **Make the prompt (Step 5):** each essay becomes one text: instructions, rubric, essay, and the correct scores.
5. **Train LLaMA-3-8B (Steps 6–7):** it learns to write the right scores. This is the slow module.

| | Fast module | Slow module |
|---|---|---|
| **How it scores** | A small classifier reads the model's numbers for the essay | The model writes the scores as text |
| **Speed** | Under 1 second | A few seconds |
| **When it is used** | Always, first | Only when the fast module is not sure (confidence below 0.2) |

6. **Test the slow module (Step 8):** the model writes scores for an essay.
7. **Turn essays into numbers (Step 9):** the model changes each essay into 4,096 numbers (an embedding).
8. **Train the fast module (Step 10):** one small classifier per trait learns scores from those numbers.
9. **Confidence (Step 11):** for each score, 0 = just guessing, 1 = completely sure.
10. **Combine (Step 12):** keep the fast score; if confidence is below 0.2, ask the slow module.
11. **Measure (Steps 13–14):** compare our scores with the teachers' using QWK (1 = same as the teachers, 0 = random). The paper got 0.758 (fast) and 0.761 (both) for the overall score.
12. **Save (Step 15):** the trained model goes to Google Drive for the website.

## 3. How the website works

1. The user pastes an essay and clicks **Score essay**.
2. The page sends the essay to the backend.
3. The backend checks that the essay has at least 10 words.
4. The fast module scores every trait; the slow module re-checks only the traits it is unsure about.
5. The page shows a table: score, confidence, and whether the fast or slow module gave it.

The model needs a GPU, so the website runs on Colab and ngrok gives it a public link. The code to start it is in the README.

## 4. Files in the repo

| File | What it does |
|---|---|
| `notebooks/dual_process_aes.ipynb` | Trains and tests the model |
| `app/predictor.py` | Loads the trained model and scores an essay |
| `app/main.py` | The backend: receives the essay and returns the scores |
| `app/static/` | The web page (HTML, CSS, JavaScript) |
| `requirements.txt` | The Python libraries to install |
| `README.md` | How to train and how to run the website |
| `data/`, `models/` | The dataset and trained model; kept in Google Drive, not on GitHub |