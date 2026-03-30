"""
Shared data for the Mood Machine lab.

This file defines:
  - POSITIVE_WORDS: starter list of positive words
  - NEGATIVE_WORDS: starter list of negative words
  - SAMPLE_POSTS: short example posts for evaluation and training
  - TRUE_LABELS: human labels for each post in SAMPLE_POSTS
"""

# ---------------------------------------------------------------------
# Starter word lists
# ---------------------------------------------------------------------

POSITIVE_WORDS = [
    "happy",
    "great",
    "good",
    "love",
    "excited",
    "awesome",
    "fun",
    "chill",
    "relaxed",
    "amazing",
]

NEGATIVE_WORDS = [
    "sad",
    "bad",
    "terrible",
    "awful",
    "angry",
    "upset",
    "tired",
    "stressed",
    "hate",
    "boring",
    "anxiety",
]

# ---------------------------------------------------------------------
# Starter labeled dataset
# ---------------------------------------------------------------------

# Short example posts written as if they were social media updates or messages.
SAMPLE_POSTS = [
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful", #ambiguous/mixed 
    "This is fine",
    "So excited for the weekend",
    "I am not happy about this",
    "Can't wait to go out tonight",
    "I need to lock in",
    "I am always nonchalant about everything",
    "That is tuff but I am not going to let it get to me",
    "I almost crashed out reading that code",
    "People vibe-coded this and it is not accurate",
    "You look good 🫣", #ambiguous
    "It is giving me anxiety", #ambiguous
    "I love getting stuck in traffic", # sarcasm
    "Great, another bug in production", # sarcasm
    "Just what I needed, more delays", # sarcasm
    "Love that for me 🙄", # sarcasm
    "Amazing, my laptop crashed again", # sarcasm

]

# Human labels for each post above.
# Allowed labels in the starter:
#   - "positive"
#   - "negative"
#   - "neutral"
#   - "mixed"
TRUE_LABELS = [
    "positive",  # "I love this class so much"
    "negative",  # "Today was a terrible day"
    "mixed",     # "Feeling tired but kind of hopeful"
    "neutral",   # "This is fine"
    "positive",  # "So excited for the weekend"
    "negative",  # "I am not happy about this"
    "positive", # "Can't wait to go out tonight"
    "positive", # "I need to lock in"
    "neutral", # "I am always nonchalant about everything"
    "positive", # "That is tuff but I am not going to let it get to me"
    "negative", # "I almost crashed out reading that code"
    "negative", # "People vibe-coded this and it is not accurate"
    "mixed", # "You look good 🫣"
    "negative", # "It is giving me anxiety"
    "negative", # "I love getting stuck in traffic"
    "negative", # "Great, another bug in production"
    "negative", # "Just what I needed, more delays"
    "negative", # "Love that for me 🙄"
    "negative", # "Amazing, my laptop crashed again"
]