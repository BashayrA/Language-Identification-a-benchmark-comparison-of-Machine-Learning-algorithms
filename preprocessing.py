import re

# ----------------------------------------------------------------------------
# Preprocessing helpers - identical logic to the original notebook.
# Shared by train_model.py (training) and app.py (inference) so both stay
# perfectly in sync.
# ----------------------------------------------------------------------------

URL_PATTERN = re.compile(r"http\S+")

# Broad unicode ranges covering common emoji blocks (stand-in for the `emoji`
# package used in the notebook, avoids an extra dependency)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002B00-\U00002BFF"
    "]+",
    flags=re.UNICODE,
)


def contains_url(text: str) -> bool:
    return bool(URL_PATTERN.search(text))


def contains_emoji(text: str) -> bool:
    return bool(EMOJI_PATTERN.search(text))


def clean_text(text: str) -> str:
    """Identical to the notebook's clean_text(): strip + lowercase."""
    text = str(text)
    text = text.strip()
    text = text.lower()
    return text