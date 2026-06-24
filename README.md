# TakeMeter

A fine-tuned text classifier that evaluates discourse quality in r/LetsTalkMusic, a Reddit community dedicated to serious music discussion.

---

## Community Choice

r/LetsTalkMusic is a subreddit for in-depth music discussion. Unlike general music communities, it attracts listeners who want to go beyond surface-level reactions and engage seriously with albums, artists, and genres. The discourse varies enormously — some posts are detailed analytical arguments, some are bare opinions with no supporting reasoning, and many are simply questions seeking information or community experience. This variation makes it a strong candidate for a classification task: the distinctions are real and meaningful to community members, and the labels map onto genuinely different types of contributions.

---

## Label Taxonomy

Three labels were defined to capture the primary modes of discourse in the community:

**opinion** — The post makes a claim or expresses a view without substantial supporting reasoning or evidence. The post may assert something strongly, but does not explain why beyond personal feeling or brief assertion.

- Example 1: *"Insane Clown Posse is actually kind of underrated. More than Nickelback, more than Imagine Dragons, more than Limp Bizkit. Unlike those 3, ICP NEVER had support from media, and their fans were designated a gang."*
- Example 2: *"Crunkcore was overhated. I think crunkcore gets way too much hate, it's genuinely not that bad."*

**analysis** — The post makes a claim and backs it with specific evidence, examples, comparisons, or developed reasoning. The post goes beyond assertion to explain *why* something is true.

- Example 1: *"Why songs are getting shorter has nothing to do with streaming and everything to do with the skip button. Before skipping was instant you sat with things, not because you were more patient, because abandoning a track mid song cost something. The skip button made patience optional and most people stopped exercising it."*
- Example 2: *"Everyone talks about how grunge killed hair metal but nobody seems to talk about how hair metal did this to AOR bands like Journey/Foreigner/REO Speedwagon. By like 1985 these bands all became adult contemporary, basically. Like Celine Dion with guitars."*

**question** — The post seeks information, recommendations, or community experiences. It contains no central claim and is oriented toward receiving rather than making an argument.

- Example 1: *"What's the best way to browse through genres? I tried browsing through genres on rateyourmusic but some genres there have a ton of subgenres making it too much work sometimes."*
- Example 2: *"How did you 'discover' your favorite bands?"*

---

## Data Collection

**Source:** Posts were collected manually from r/LetsTalkMusic by browsing the subreddit's hot, top, and new feeds and copying post titles and body text into a spreadsheet.

**Labeling process:** Posts were pre-labeled using Claude (claude-sonnet-4-6) with label definitions provided in the prompt. All 200 pre-assigned labels were then reviewed and corrected manually. Labels were assigned based on the definitions above — the key criterion for distinguishing opinion from analysis was whether the post contained developed reasoning (more than one supporting claim or a specific example used as evidence).

**Label distribution:**

| Label    | Count | Percentage |
|----------|-------|------------|
| question | 103   | 51.5%      |
| analysis | 56    | 28.0%      |
| opinion  | 41    | 20.5%      |
| **Total**| **200** | **100%** |

**Difficult-to-label examples:**

1. *"Are any of Arcade Fire's albums significant?"* — The title is a question, but the body makes a clear opinion: "The Suburbs sounds to me like the sonic equivalent of Stranger Things... their music just feels empty." Decided: **opinion**, because the primary thrust of the post is expressing a view, not seeking information.

2. *"Is the loss of a monoculture better or worse for artists?"* — Presents developed reasoning for both sides, which resembles analysis. However, the post never takes a position — it is genuinely asking the community to weigh in. Decided: **question**, because no central claim is made.

3. *"How music was before"* — No question mark, no explicit stated claim, but clearly not a question. The post implicitly argues that streaming stripped context from music through developed personal reasoning. Decided: **analysis**, because the reasoning is developed even if the thesis is implicit rather than stated.

---

## Fine-Tuning Approach

**Base model:** `distilbert-base-uncased`

**Training setup:** Fine-tuned using HuggingFace Transformers on Google Colab with a T4 GPU. Dataset split: 70% train (140 examples), 15% validation (30 examples), 15% test (30 examples).

**Default hyperparameters used:** 3 epochs, learning rate 2e-5, batch size 16. A second run with 5 epochs was attempted after observing flat validation accuracy — performance did not improve, confirming the issue was not epoch count.

**Key hyperparameter decision:** The learning rate of 2e-5 was kept at the default. Given that validation accuracy was flat from epoch 1 onward, the issue was not learning rate sensitivity but rather insufficient training signal — too few examples in the minority classes for distilbert to learn the opinion/analysis boundary.

---

## Baseline

**Model:** Groq `llama-3.3-70b-versatile`, zero-shot (no task-specific training).

**Prompt used:**

```
You are classifying posts from r/LetsTalkMusic, a music discussion subreddit.
Assign each post to exactly one of the following categories.

opinion: The post makes a claim or expresses a view without substantial supporting reasoning or evidence.
Example: "Insane Clown Posse is actually kind of underrated. More than Nickelback, more than Imagine Dragons..."

analysis: The post makes a claim and backs it with specific evidence, examples, comparisons, or developed reasoning.
Example: "Why songs are getting shorter has nothing to do with streaming and everything to do with the skip button..."

question: The post seeks information, recommendations, or community experiences. It contains no central claim.
Example: "What's the best way to browse through genres? I tried browsing through genres on rateyourmusic..."

Respond with ONLY the label name.
Do not explain your reasoning.

Valid labels:
opinion
analysis
question
```

**How results were collected:** The notebook classified all 30 test examples using the Groq API. All 30 responses were parseable (0 unparseable).

---

## Evaluation Report

### Overall Accuracy

| Model         | Accuracy |
|---------------|----------|
| Baseline (Llama-3.3-70b, zero-shot) | **76.7%** |
| Fine-tuned (distilbert-base-uncased) | **53.3%** |

The fine-tuned model performed worse than the zero-shot baseline by 23 percentage points.

### Per-Class Metrics — Baseline

| Label    | Precision | Recall | F1   | Support |
|----------|-----------|--------|------|---------|
| opinion  | 0.86      | 1.00   | 0.92 | 6       |
| analysis | 1.00      | 0.33   | 0.50 | 9       |
| question | 0.70      | 0.93   | 0.80 | 15      |
| **macro avg** | 0.85 | 0.76  | 0.74 | 30      |

### Per-Class Metrics — Fine-Tuned Model

| Label    | Precision | Recall | F1   | Support |
|----------|-----------|--------|------|---------|
| opinion  | 0.00      | 0.00   | 0.00 | 6       |
| analysis | 1.00      | 0.11   | 0.20 | 9       |
| question | 0.52      | 1.00   | 0.68 | 15      |
| **macro avg** | 0.51 | 0.37  | 0.29 | 30      |

### Confusion Matrix — Fine-Tuned Model

|                  | Predicted: opinion | Predicted: analysis | Predicted: question |
|------------------|--------------------|---------------------|---------------------|
| **True: opinion**   | 0                  | 0                   | 6                   |
| **True: analysis**  | 0                  | 1                   | 8                   |
| **True: question**  | 0                  | 0                   | 15                  |

The model predicted `question` for 29 of 30 examples. It correctly identified 1 analysis post and all 15 question posts, but failed entirely on opinion.

### Analysis of Wrong Predictions

**Which labels are being confused?** The confusion is almost entirely unidirectional: opinion → question and analysis → question. The model never predicts `opinion` at all, and predicts `analysis` only once. Every misclassification is a collapse into the majority class.

**Why is this boundary hard?** The `opinion` vs `analysis` distinction requires understanding the *degree* of reasoning in a post — a judgment that depends on semantic content rather than surface features. Distilbert, with only 41 opinion examples and 56 analysis examples in training, did not have enough signal to learn this boundary. Posts that look structurally similar (a claim followed by one or two sentences) were consistently sent to `question`, suggesting the model learned question as a safe default rather than learning any of the substantive distinctions.

**Is this a labeling or data problem?** Both. The opinion/analysis boundary is genuinely subjective — two annotators would disagree on many edge cases — which introduces noise that a small model cannot overcome. Additionally, 140 training examples split across three classes (with only ~29 opinion examples in training) is insufficient for distilbert to learn a nuanced distinction.

**What would fix it?** More examples — particularly more opinion and analysis examples — and potentially a tighter definition of the opinion/analysis boundary with explicit rules about what counts as "developed reasoning."

**Three specific wrong predictions:**

> **[YOU FILL IN: Go to your Section 4 output in Colab, find 3 specific posts the model got wrong, and paste them here with the true label, predicted label, and a sentence of analysis. Example format below — replace with your actual examples.]**

1. **Post:** *[paste the post text]* | **True label:** analysis | **Predicted:** question | **Why it failed:** [your analysis]

2. **Post:** *[paste the post text]* | **True label:** opinion | **Predicted:** question | **Why it failed:** [your analysis]

3. **Post:** *[paste the post text]* | **True label:** [true] | **Predicted:** [predicted] | **Why it failed:** [your analysis]

### Sample Classifications

> **[YOU FILL IN: Run 3–5 posts through your fine-tuned model in Colab and record the predicted label and confidence score for each. Use the table format below.]**

| Post (truncated) | True Label | Predicted Label | Confidence |
|-----------------|------------|-----------------|------------|
| [paste post excerpt] | question | question | 0.XX |
| [paste post excerpt] | analysis | question | 0.XX |
| [paste post excerpt] | opinion | question | 0.XX |

For the correctly predicted example: [one sentence explaining why the prediction is reasonable].

---

## Reflection: What the Model Learned vs. What I Intended

I intended the model to learn the distinction between three types of discourse contributions — questions seeking information, opinions asserting without evidence, and analysis making claims with reasoning. What the model actually learned was a much simpler heuristic: predict `question` for almost everything.

This reveals that the distinctions I cared about — the presence or absence of developed reasoning — are not legible to a small model from 140 training examples. The `opinion` and `analysis` labels require understanding the *quality* of argumentation in a post, which is a higher-order semantic judgment that distilbert was not able to acquire with this dataset size.

The zero-shot baseline (Llama-3.3-70b) performed significantly better precisely because it already understands language at a level that lets it apply the label definitions directly. Fine-tuning distilbert would require substantially more data and potentially simpler, more surface-distinguishable labels to produce a model that outperforms a capable zero-shot baseline.

---

## Spec Reflection

**One way the spec helped:** The requirement to define hard edge cases before annotating forced me to sharpen the opinion/analysis boundary early. Writing down the rule ("analysis requires developed reasoning, not just one sentence of support") made annotation more consistent and gave me a concrete definition to put in the Groq prompt.

**One way implementation diverged:** The spec assumes the fine-tuned model will outperform the baseline, and the evaluation section is structured around documenting how much improvement fine-tuning provided. In this project, the fine-tuned model performed worse. Rather than treating this as a failure to hide, I documented it honestly and used it as the central finding: the task as defined is too subjective and data-sparse for distilbert to learn, while a large zero-shot model handles it well. This divergence from the expected outcome is itself the most informative result of the project.

---

## AI Usage

1. **Pre-labeling the dataset:** I provided Claude (claude-sonnet-4-6) with the 200 unlabeled posts and my label definitions, and asked it to assign one label per post. Claude produced a labeled CSV. I then reviewed every label manually and corrected cases where I disagreed — primarily on the opinion/analysis boundary, where Claude tended to assign `analysis` to posts with any visible reasoning. I overrode approximately [YOU FILL IN: estimate how many you changed] labels during review.

2. **Label taxonomy design:** I used Claude to stress-test the label definitions before annotating. I described the three labels and asked Claude to identify which would be hardest to apply consistently. Claude flagged the opinion/analysis boundary as the most problematic, noting that any post with one sentence of reasoning sits ambiguously between the two. I used this to tighten my definition: analysis requires *developed* reasoning, not just a single supporting sentence.

3. **Error pattern analysis:** After receiving the wrong predictions from Section 4, I provided them to Claude and asked it to identify common themes. Claude identified the unidirectional collapse to `question` as the dominant pattern, consistent with the confusion matrix. I verified this independently by reading the misclassified examples.

---

## Dataset

The labeled dataset is included in this repository as `takemeter_labeled.csv`. Columns: `text` (post title + body), `label` (opinion / analysis / question), `notes` (annotation decisions for edge cases).
