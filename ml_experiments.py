from typing import List, Tuple

try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: scikit-learn.\n"
        "Install it in this project environment with one of these commands:\n"
        "  source .venv/bin/activate && pip install -r requirements.txt\n"
        "  /Users/villegasjb/Documents/GitHub/ai110-module3tinker-themoodmachine-starter/.venv/bin/python -m pip install scikit-learn"
    ) from exc

from dataset import SAMPLE_POSTS, TRUE_LABELS


def train_ml_model(
    texts: List[str],
    labels: List[str],
) -> Tuple[CountVectorizer, LogisticRegression]:
    
    if len(texts) != len(labels):
        raise ValueError(
            "texts and labels must be the same length. "
            "Check SAMPLE_POSTS and TRUE_LABELS in dataset.py."
        )

    if not texts:
        raise ValueError("No training data provided. Add examples in dataset.py.")

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, labels)

    return vectorizer, model


def evaluate_on_dataset(
    texts: List[str],
    labels: List[str],
    vectorizer: CountVectorizer,
    model: LogisticRegression,
) -> float:
    
    if len(texts) != len(labels):
        raise ValueError(
            "texts and labels must be the same length. "
            "Check your dataset."
        )

    X = vectorizer.transform(texts)
    preds = model.predict(X)

    print("=== ML Model Evaluation on Dataset ===")
    correct = 0
    for text, true_label, pred_label in zip(texts, labels, preds):
        is_correct = pred_label == true_label
        if is_correct:
            correct += 1
        print(f'"{text}" -> predicted={pred_label}, true={true_label}')

    accuracy = accuracy_score(labels, preds)
    print(f"\nAccuracy on this dataset: {accuracy:.2f}")
    return accuracy


def predict_single_text(
    text: str,
    vectorizer: CountVectorizer,
    model: LogisticRegression,
) -> str:
    """
    Predict the mood label for a single text string using
    the trained ML model.
    """
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    return pred


def run_interactive_loop(
    vectorizer: CountVectorizer,
    model: LogisticRegression,
) -> None:
   
    print("\n=== Interactive Mood Machine (ML model) ===")
    print("Type a sentence to analyze its mood.")
    print("Type 'quit' or press Enter on an empty line to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input == "" or user_input.lower() == "quit":
            print("Goodbye from the ML Mood Machine.")
            break

        label = predict_single_text(user_input, vectorizer, model)
        print(f"ML model: {label}")


if __name__ == "__main__":
    print("Training an ML model on SAMPLE_POSTS and TRUE_LABELS from dataset.py...")
    print("Make sure you have added enough labeled examples before running this.\n")

    # Train the model on the current dataset.
    vectorizer, model = train_ml_model(SAMPLE_POSTS, TRUE_LABELS)

    # Evaluate on the same dataset (training accuracy).
    evaluate_on_dataset(SAMPLE_POSTS, TRUE_LABELS, vectorizer, model)

    # Let the user try their own examples.
    run_interactive_loop(vectorizer, model)

    print("\nTip: Compare these predictions with the rule based model")
    print("by running `python main.py`. Notice where they fail in")
    print("similar ways and where they fail in different ways.")
