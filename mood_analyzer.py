# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re
from typing import List, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    NEGATION_WORDS = {"not", "never", "no", "cannot"}
    INTENSIFIERS = {"very": 1.5, "really": 1.5, "super": 1.7}
    DOWNTONERS = {"slightly": 0.5, "kinda": 0.6}
    CONTRAST_WORDS = {"but", "however"}
    NEGATION_WINDOW = 3
    NEGATION_SKIP_WORDS = {"very", "really", "super", "slightly", "kinda", "kind", "of", "at", "all", "just"}
    WORD_WEIGHT_OVERRIDES = {
        "amazing": 2.0,
        "awesome": 1.5,
        "terrible": -2.0,
        "awful": -2.0,
        "hate": -1.5,
        "annoyed": -2.0,
    }
    EMOJI_SLANG_WEIGHTS = {
        ":)": 2.0,
        ":(": -2.0,
        "lol": 1.5,
        "lmao": 2.0,
        "💀": -2.0,
        "😭": -2.0,
    }
    SARCASM_PHRASE_OVERRIDES = {
        "love getting stuck in traffic": -4.0,
        "great another bug": -3.0,
        "great another bug in production": -3.5,
        "just what i needed": -2.5,
        "love that for me": -2.5,
        "amazing my laptop crashed again": -4.0,
    }
    POSITIVE_CUES = {"love", "great", "amazing", "awesome", "good"}
    NEGATIVE_CONTEXT_WORDS = {
        "traffic",
        "stuck",
        "delay",
        "delays",
        "waiting",
        "bug",
        "crash",
        "crashed",
        "crashes",
        "broke",
        "broken",
        "error",
        "errors",
    }

    # Map emoji/slang patterns to sentiment-bearing tokens.
    EMOJI_SLANG_MAP = {
        ":)": " happy ",
        ":(": " sad ",
        "lol": " happy ",
        "lmao": " happy ",
        "💀": " sad ",
        "😭": " sad ",
    }

    CONTRACTION_MAP = {
        "can't": "can not",
        "cannot": "can not",
        "won't": "will not",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "haven't": "have not",
        "hasn't": "has not",
        "hadn't": "had not",
        "couldn't": "could not",
        "shouldn't": "should not",
        "wouldn't": "would not",
        "mustn't": "must not",
        "n't": " not",
    }

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        
        cleaned = text.strip().lower().replace("’", "'")

        # Expand common contractions before tokenization to expose negation.
        cleaned = self._expand_contractions(cleaned)

        # Replace simple emoji/slang markers with sentiment-bearing tokens.
        cleaned = self._map_emoji_and_slang(cleaned)

        # Keep letters, digits, spaces, and apostrophes, and replace other
        # characters with spaces so punctuation does not stick to words.
        cleaned = re.sub(r"[^a-z0-9'\s]", " ", cleaned)

        # Simple tokenization: collect word-like chunks, then drop tokens that
        # become empty after trimming surrounding apostrophes.
        raw_tokens = re.findall(r"[a-z0-9']+", cleaned)
        normalized_tokens: List[str] = []
        for token in raw_tokens:
            stripped = token.strip("'")
            if not stripped:
                continue
            normalized_tokens.append(self._normalize_and_lemmatize(stripped))

        return normalized_tokens
    
    #STEMMING: rule based process that trims word endings to produce root form (stem)
    #LEMMATIZATION: converts words to their dictionary form (lemma) considers - grammar, vocab, and context - and it is context aware.

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> float:
 
        tokens = self.preprocess(text)
        raw_text = text.lower()
        normalized_raw_text = re.sub(r"[^a-z0-9\s]", " ", raw_text)
        normalized_raw_text = re.sub(r"\s+", " ", normalized_raw_text).strip()

        contrast_index = next(
            (idx for idx, tok in enumerate(tokens) if tok in self.CONTRAST_WORDS),
            -1,
        )

        # Start from 0, add positive signals, subtract negative signals.
        score = 0.0

        # Phrase overrides have highest priority for clear sarcasm patterns.
        for phrase, phrase_weight in self.SARCASM_PHRASE_OVERRIDES.items():
            occurrences = normalized_raw_text.count(phrase)
            if occurrences:
                score += phrase_weight * occurrences

        i = 0

        while i < len(tokens):
            token = tokens[i]

            if token in self.CONTRAST_WORDS:
                i += 1
                continue

            clause_multiplier = 1.0
            if contrast_index != -1:
                clause_multiplier = 0.7 if i < contrast_index else 1.3

            # Base token weight. If no override exists, use +/-1 polarity.
            base_weight = self.WORD_WEIGHT_OVERRIDES.get(
                token,
                float(self._token_polarity(token)),
            )

            # Intensifier and downtoner handling for adjacent sentiment tokens.
            modifier_multiplier = 1.0
            if token in self.INTENSIFIERS and i + 1 < len(tokens):
                next_token = tokens[i + 1]
                next_weight = self.WORD_WEIGHT_OVERRIDES.get(
                    next_token,
                    float(self._token_polarity(next_token)),
                )
                if next_weight != 0:
                    score += next_weight * self.INTENSIFIERS[token] * clause_multiplier
                    i += 2
                    continue

            if token in self.DOWNTONERS and i + 1 < len(tokens):
                next_token = tokens[i + 1]
                next_weight = self.WORD_WEIGHT_OVERRIDES.get(
                    next_token,
                    float(self._token_polarity(next_token)),
                )
                if next_weight != 0:
                    score += next_weight * self.DOWNTONERS[token] * clause_multiplier
                    i += 2
                    continue

            # Negation window: scan ahead for the next sentiment-bearing token.
            if token in self.NEGATION_WORDS and i + 1 < len(tokens):
                j = i + 1
                checked = 0
                while j < len(tokens) and checked < self.NEGATION_WINDOW:
                    candidate = tokens[j]
                    if candidate in self.NEGATION_SKIP_WORDS:
                        j += 1
                        continue

                    checked += 1
                    candidate_weight = self.WORD_WEIGHT_OVERRIDES.get(
                        candidate,
                        float(self._token_polarity(candidate)),
                    )
                    if candidate_weight != 0:
                        local_multiplier = 1.0
                        if j - 1 >= i + 1:
                            maybe_modifier = tokens[j - 1]
                            if maybe_modifier in self.INTENSIFIERS:
                                local_multiplier = self.INTENSIFIERS[maybe_modifier]
                            elif maybe_modifier in self.DOWNTONERS:
                                local_multiplier = self.DOWNTONERS[maybe_modifier]

                        score -= candidate_weight * local_multiplier * clause_multiplier
                        i = j + 1
                        break

                    j += 1

                if i == j + 1:
                    continue

            # Count-based scoring: every occurrence contributes.
            score += base_weight * modifier_multiplier * clause_multiplier
            i += 1

        # Strong emoji/slang signals from the raw text.
        for marker, marker_weight in self.EMOJI_SLANG_WEIGHTS.items():
            occurrences = raw_text.count(marker)
            if occurrences:
                score += marker_weight * occurrences

        # Positive-plus-negative-context sarcasm rule:
        # if positive cue appears close to negative context, add a penalty.
        sarcasm_penalties = 0
        for idx, token in enumerate(tokens):
            if token not in self.POSITIVE_CUES:
                continue

            left = max(0, idx - 4)
            right = min(len(tokens), idx + 5)
            window = tokens[left:right]
            if any(t in self.NEGATIVE_CONTEXT_WORDS for t in window):
                sarcasm_penalties += 1

        if sarcasm_penalties:
            score -= 2.5 * sarcasm_penalties

        return score
    
    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        
        score = self.score_text(text)
        if score >= 1.0:
            return "positive"
        elif score <= -1.0:
            return "negative"
        elif score == 0:
            return "neutral"
        return "mixed"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        tokens = self.preprocess(text)
        raw_text = text.lower()
        normalized_raw_text = re.sub(r"[^a-z0-9\s]", " ", raw_text)
        normalized_raw_text = re.sub(r"\s+", " ", normalized_raw_text).strip()

        contrast_index = next(
            (idx for idx, tok in enumerate(tokens) if tok in self.CONTRAST_WORDS),
            -1,
        )

        score = 0.0
        positive_hits: List[str] = []
        negative_hits: List[str] = []
        negation_hits: List[str] = []
        contributions: List[str] = []

        for phrase, phrase_weight in self.SARCASM_PHRASE_OVERRIDES.items():
            occurrences = normalized_raw_text.count(phrase)
            if occurrences:
                delta = phrase_weight * occurrences
                score += delta
                contributions.append(
                    f"sarcasm_phrase('{phrase}') x{occurrences} -> {delta:+.2f}"
                )

        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token in self.CONTRAST_WORDS:
                i += 1
                continue

            clause_multiplier = 1.0
            if contrast_index != -1:
                clause_multiplier = 0.7 if i < contrast_index else 1.3

            base_weight = self.WORD_WEIGHT_OVERRIDES.get(
                token,
                float(self._token_polarity(token)),
            )

            if token in self.INTENSIFIERS and i + 1 < len(tokens):
                next_token = tokens[i + 1]
                next_weight = self.WORD_WEIGHT_OVERRIDES.get(
                    next_token,
                    float(self._token_polarity(next_token)),
                )
                if next_weight != 0:
                    delta = next_weight * self.INTENSIFIERS[token] * clause_multiplier
                    score += delta
                    contributions.append(
                        f"intensifier('{token} {next_token}') -> {delta:+.2f}"
                    )
                    if delta > 0:
                        positive_hits.append(f"{token} {next_token}")
                    else:
                        negative_hits.append(f"{token} {next_token}")
                    i += 2
                    continue

            if token in self.DOWNTONERS and i + 1 < len(tokens):
                next_token = tokens[i + 1]
                next_weight = self.WORD_WEIGHT_OVERRIDES.get(
                    next_token,
                    float(self._token_polarity(next_token)),
                )
                if next_weight != 0:
                    delta = next_weight * self.DOWNTONERS[token] * clause_multiplier
                    score += delta
                    contributions.append(
                        f"downtoner('{token} {next_token}') -> {delta:+.2f}"
                    )
                    if delta > 0:
                        positive_hits.append(f"{token} {next_token}")
                    else:
                        negative_hits.append(f"{token} {next_token}")
                    i += 2
                    continue

            if token in self.NEGATION_WORDS and i + 1 < len(tokens):
                j = i + 1
                checked = 0
                while j < len(tokens) and checked < self.NEGATION_WINDOW:
                    candidate = tokens[j]
                    if candidate in self.NEGATION_SKIP_WORDS:
                        j += 1
                        continue

                    checked += 1
                    candidate_weight = self.WORD_WEIGHT_OVERRIDES.get(
                        candidate,
                        float(self._token_polarity(candidate)),
                    )
                    if candidate_weight != 0:
                        local_multiplier = 1.0
                        if j - 1 >= i + 1:
                            maybe_modifier = tokens[j - 1]
                            if maybe_modifier in self.INTENSIFIERS:
                                local_multiplier = self.INTENSIFIERS[maybe_modifier]
                            elif maybe_modifier in self.DOWNTONERS:
                                local_multiplier = self.DOWNTONERS[maybe_modifier]

                        delta = -candidate_weight * local_multiplier * clause_multiplier
                        score += delta
                        phrase = f"{token} ... {candidate}"
                        negation_hits.append(phrase)
                        contributions.append(f"negation('{phrase}') -> {delta:+.2f}")
                        if delta > 0:
                            positive_hits.append(phrase)
                        else:
                            negative_hits.append(phrase)
                        i = j + 1
                        break

                    j += 1

                if i == j + 1:
                    continue

            delta = base_weight * clause_multiplier
            score += delta
            if delta != 0:
                contributions.append(f"token('{token}') -> {delta:+.2f}")
                if delta > 0:
                    positive_hits.append(token)
                else:
                    negative_hits.append(token)
            i += 1

        for marker, marker_weight in self.EMOJI_SLANG_WEIGHTS.items():
            occurrences = raw_text.count(marker)
            if occurrences:
                delta = marker_weight * occurrences
                score += delta
                contributions.append(f"emoji_slang('{marker}') x{occurrences} -> {delta:+.2f}")
                if delta > 0:
                    positive_hits.append(marker)
                else:
                    negative_hits.append(marker)

        sarcasm_penalties = 0
        for idx, token in enumerate(tokens):
            if token not in self.POSITIVE_CUES:
                continue

            left = max(0, idx - 4)
            right = min(len(tokens), idx + 5)
            window = tokens[left:right]
            if any(t in self.NEGATIVE_CONTEXT_WORDS for t in window):
                sarcasm_penalties += 1

        if sarcasm_penalties:
            delta = -2.5 * sarcasm_penalties
            score += delta
            contributions.append(f"sarcasm_context_penalty x{sarcasm_penalties} -> {delta:+.2f}")
            negative_hits.append("positive-cue near negative-context")

        model_score = self.score_text(text)
        if abs(model_score - score) > 1e-9:
            contributions.append(
                f"debug_score_mismatch(explain={score:+.2f}, score_text={model_score:+.2f})"
            )
            score = model_score

        return (
            f"Score = {score:.2f} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'}, "
            f"negations: {negation_hits or '[]'}, "
            f"contributions: {contributions or '[]'})"
        )

    def _expand_contractions(self, text: str) -> str:
        expanded = text
        for contraction, replacement in self.CONTRACTION_MAP.items():
            expanded = re.sub(rf"\b{re.escape(contraction)}\b", replacement, expanded)
        return expanded

    def _map_emoji_and_slang(self, text: str) -> str:
        mapped = text
        for pattern, replacement in self.EMOJI_SLANG_MAP.items():
            mapped = re.sub(rf"{re.escape(pattern)}", replacement, mapped)
        return mapped

    def _normalize_and_lemmatize(self, token: str) -> str:
        # Elongation normalization: cap repeated chars at 2 (soooo -> soo).
        normalized = re.sub(r"(.)\1{2,}", r"\1\1", token)
        return self._lemmatize_token(normalized)

    def _lemmatize_token(self, token: str) -> str:
        if token in self.positive_words or token in self.negative_words:
            return token

        candidates = [token]
        if token.endswith("ies") and len(token) > 4:
            candidates.append(token[:-3] + "y")
        if token.endswith("ied") and len(token) > 4:
            candidates.append(token[:-3] + "y")
        if token.endswith("ing") and len(token) > 5:
            base = token[:-3]
            candidates.extend([base, base + "e"])
            if len(base) > 2 and base[-1] == base[-2]:
                candidates.append(base[:-1])
        if token.endswith("ed") and len(token) > 4:
            base = token[:-2]
            candidates.extend([base, base + "e"])
            if len(base) > 2 and base[-1] == base[-2]:
                candidates.append(base[:-1])
        if token.endswith("s") and len(token) > 3:
            candidates.append(token[:-1])

        for candidate in candidates:
            if candidate in self.positive_words or candidate in self.negative_words:
                return candidate
        return token

    def _token_polarity(self, token: str) -> int:
        if token in self.positive_words:
            return 1
        if token in self.negative_words:
            return -1
        return 0
