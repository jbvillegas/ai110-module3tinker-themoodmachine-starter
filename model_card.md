# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn

You may complete this model card for whichever version you used, or compare both if you explored them.

## 1. Model Overview

**Model type:**  
Describe whether you used the rule based model, the ML model, or both.  
Example: “I used the rule based model only” or “I compared both models.”

- I used both models, the rule based model and the ML model and I compared them.

**Intended purpose:**  
What is this model trying to do?  
Example: classify short text messages as moods like positive, negative, neutral, or mixed.

- The model is designed to classify short, social text kind of like expression from humans into one of four mood categories, which are the following: positive, negative, neutral, or mixed. 

- Moreover, the goal of the model is to help us learn how rule logic compares with a learned model on the exact same dataset. 

**How it works (brief):**  
For the rule based version, describe the scoring rules you created.  
For the ML version, describe how training works at a high level (no math needed).

- The rule based model works in the following manner: 
    1. The text gets normalized (lowercarse, punctuation cleanup, tokenization, normalization)
    2. Tokens are assigned a score using sentiment lexicons and handrafted rules.
    3. The end numeric score gets mapped to a pre-determined label.

- The machine learning model works in the following manner: 
    1. The text gets converted into multiple vectors using the function CountVectorizer.
    2. The LogisticRegression model gets trained on the sample data "SAMPLE_POSTS" and "TRUE_LABELS"
    3. The model learns the relationships between the tokens and the labels from the examples provided rather than the specific rules. 

## 2. Data

**Dataset description:**  
Summarize how many posts are in `SAMPLE_POSTS` and how you added new ones.

- The dataset has a total of 19 labeled short posts. Initially, we were provided with examples and I expanded the dataset with difficult cases, mostly cases focused to sarcasm. 

**Labeling process:**  
Explain how you chose labels for your new examples.  
Mention any posts that were hard to label or could have multiple valid labels.

- I expanded SAMPLE_POSTS in the following way:

    1. I added 5 sarcasm-oriented examples to the stress-test failures where literal positive words hide negative intent such as:
        "I love getting stuck in traffic"
        "Great, another bug in production"
        "Just what I needed, more delays"
        "Love that for me 🙄"
        "Amazing, my laptop crashed again"
    
    2. I modified the labeling process by labeling each post based on intended emotional meanining, not just the literal keywords and its polarity. 
        Negative: sarcastic posts containing words such as "love", "great", "amazing" were labeled negative due to negative context.
        Mixed: the label mixed was added to posts that have conflicting signals such as being tired but feeling hopeful.
        Neutral: this one was kept for low-signal statements that do not have a clear positive or negative mood. 

- Hard to label examples: 

    1. "You look good 🫣", this can be interpreted as positive, flirty, awkward, or ambiguous depending on context and relationship.
    2. "Feeling tired but kind of hopeful", this has both positive and negative cues; mixed is reasonable, but some annotators might still pick neutral.

**Important characteristics of your dataset:**  
Examples you might include:  

- Contains slang or emojis  
- Includes sarcasm  
- Some posts express mixed feelings  
- Contains short or ambiguous messages

- Some important characteristics of my dataset are the following: 
    1. Contains informal language and slang such as: lock in, tuff, vibe-coded.
    2. Contains emojis and expressive social-media style text. 
    3. Contains sarcasm and pragmatic meaning shifts.
    4. Contains short texts with little context.

**Possible issues with the dataset:**  
Think about imbalance, ambiguity, or missing kinds of language.

- Some of the possible issues I might have with the dataset are the following: 
    1. Having a small sample can reduce robustness, making the models less accurate.
    2. Some labels are subjective which can create inconsistency. 
    3. The set is not balanced or stratified by writing style.
    4. The language coverage is pretty reduced so it is hard to generalize. 

## 3. How the Rule Based Model Works (if used)

**Your scoring rules:**  
Describe the modeling choices you made.  
Examples:  

- How positive and negative words affect score  
- Negation rules you added  
- Weighted words  
- Emoji handling  
- Threshold decisions for labels

- The rule based model scoring rules work the following way: 
    1. We have a starting score of 0 and it accumulates token-level contributions.
    2. Positive and Negative hits contribute by occurence meaning that repetition of cues matters. 
    3. Negation is handled using a short lookahead window.
    4. Intensifiers and downtoners adjust sentiment magnitude.
    5. Word-specific weights increase/decrease impact for selected terms.
    6. Emoji/slang makers add extra sentiment signal.
    7. Contrast words such as "however" rebalance the pre/post clause influence. 
    8. Sarcasm phrase overrides and positive-cue + negative context penalties are applied. 

- Label mapping system works the following way:
    1. Score >= 1.0 is positive.
    2. Score <= -1.0 is negative.
    3. Score == 0 is neutral.
    4. Otherwise it is mixed.

**Strengths of this approach:**  
Where does it behave predictably or reasonably well?

- The model behaves reasonably well it terms of its transparency by explaining each prediction using explicit rules. Moreover, it works well on patterns that were intentionally coded as well as allowing patching in specific failure models. 

**Weaknesses of this approach:**  
Where does it fail?  
Examples: sarcasm, subtlety, mixed moods, unfamiliar slang.

- The model fails or performs poorly on the following aspects:
    1. Depends on known vocabulary and patterns.
    2. Misses intent when sentiment is implied semantically but not lexically.
    3. Requires ongoing manual updates for new slang and emerging expressions. 

## 4. How the ML Model Works (if used)

**Features used:**  
Describe the representation.  
Example: “Bag of words using CountVectorizer.”

- The feature used is the following: CountVectorizer bag-of-wrods representation.

**Training data:**  
State that the model trained on `SAMPLE_POSTS` and `TRUE_LABELS`.

- The model was trained on SAMPLE_POSTS and TRUE_LABELS from dataset.py.

**Training behavior:**  
Did you observe changes in accuracy when you added more examples or changed labels?

- Initially, with the small dataset, the behavior changes a lot when a few labels or examples are added. Furthermore, after adding sarcasm examples, training-set predictions aligned better for those cases. 

**Strengths and weaknesses:**  
Strengths might include learning patterns automatically.  
Weaknesses might include overfitting to the training data or picking up spurious cues.

- The strenghts of the model are the following: 
    1. It learns associations from data without explicitly coding each rule.
    2. It can capture combinations of tokens that may be cumbersome to handcraft. 

- The weaknesses of the model are the following: 
    1. Having 1.00 training accuracy on tiny data can indicate overfitting.
    2. It is sensitive to noisy labels and class distribution shifts.
    3. There is no guarantee that it will performs similarly on unseen text. 

## 5. Evaluation

**How you evaluated the model:**  
Both versions can be evaluated on the labeled posts in `dataset.py`.  
Describe what accuracy you observed.

- Both models were checked against the labels in dataset.py. Furthermore, these are the observed metrics: 
    1. Rule-based accuracy: 0.684 on current dataset.
    2. ML training accuracy: 1.00 on the same training set.

**Examples of correct predictions:**  
Provide 2 or 3 examples and explain why they were correct.

- These are some examples of correct predictions and the explanation:
    1. "Today was a terrible day" predicted to be negative.
    2. "I am not happy about this" predicted to be negative showing negation works.
    3. "I love getting stuck in traffic" predicted to be negative, showing that it understands sarcasm. 

**Examples of incorrect predictions:**  
Provide 2 or 3 examples and explain why the model made a mistake.  
If you used both models, show how their failures differed.

- These are some examples of correct predictions and the explanation:
    1. "Can't wait to go out tonight" was predicted neutral, however this is true positive.
        - Reason: phrase-level positivity (can’t wait) not strongly encoded.
    2. "I need to lock in" was predicted neutral, however it is true positive.
        - Reason: motivational slang not mapped as positive.
    3. "People vibe-coded this and it is not accurate" was predicted neutral, however it is true negative.
        - Reason: semantic negativity from phrase intent is not fully captured by current lexicon/rules.
    4. "You look good 🫣" was predicted positive, however it is true mixed.
        - Reason: ambiguity and social context are not modeled deeply.

- Comparison of the models: 
    1. Rule model failures are easier to interpret and patch, but coverage is limited.
    2. ML model fit training data perfectly, but this may hide weak real-world generalization due to tiny training size.

## 6. Limitations

Describe the most important limitations.  
Examples:  

- The dataset is small  
- The model does not generalize to longer posts  
- It cannot detect sarcasm reliably  
- It depends heavily on the words you chose or labeled

- These are the most important limitations: 
    1. The dataset size (19 examples) is too small for robust mood modeling.
    2. There is no held-out test split, so evaluation is optimistic (especially for ML).
    3. Sarcasm handling is still heuristic and incomplete.
    4. Context and discourse-level meaning are mostly absent.
    5. System is optimized for short informal English only.

- Example: A sentence like "You look good 🫣" can vary by context, tone, relationship, and platform norms. The model sees text only, not social context, so uncertainty is unavoidable.

## 7. Ethical Considerations

Discuss any potential impacts of using mood detection in real applications.  
Examples: 

- Misclassifying a message expressing distress  
- Misinterpreting mood for certain language communities  
- Privacy considerations if analyzing personal messages

- These are some of the potential impacts that we should consider:
    1. Distress-related misclassification may understate urgency or overstate negativity.
    2. Language-community mismatch can unfairly penalize certain slang or dialects.
    3. Privacy concerns arise when analyzing personal or sensitive messages.

- These are some of the biasses the model may currently have: 
    1. The model is tuned to the language patterns present in this small dataset.
    2. It likely performs best on short, casual English with similar slang/emoji usage.
    3. It may misinterpret users from other communities, dialects, or cultural sarcasm conventions.

## 8. Ideas for Improvement

List ways to improve either model.  
Possible directions:  

- Add more labeled data  
- Use TF IDF instead of CountVectorizer  
- Add better preprocessing for emojis or slang  
- Use a small neural network or transformer model  
- Improve the rule based scoring method  
- Add a real test set instead of training accuracy only

- These could be some possible improvements for the models: 
    1. Expand dataset substantially with more diverse writers, topics, and linguistic styles.
    2. Build train/validation/test splits to measure real generalization.
    3. Compare CountVectorizer with TF-IDF and calibrate class thresholds.
    4. Add confidence estimation and abstain behavior for low-confidence cases.
    5. Continue targeted error-driven updates (slang lexicon, sarcasm templates, phrase sentiment).
    6. Introduce phrase-level and context-aware models (for example lightweight transformer baseline) for comparison.
    7. Track annotation uncertainty for ambiguous examples instead of forcing single hard labels.
