"""Text normalization.

Two views of the input are produced:

* ``text`` -- a *safe* normalization that is what the model actually receives. Only
  NFKC folding and removal of invisible/control characters are applied, so meaning is kept.
* ``skeleton`` -- an aggressive, lossy view used only for rule matching. It lowercases,
  maps common homoglyphs and leetspeak, and collapses whitespace so trivially obfuscated
  attacks still match. It is never sent to the model.

Base64-looking tokens are decoded so instructions hidden inside an encoding can be scanned.
"""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from dataclasses import dataclass, field

_KEEP_CONTROL = {"\n", "\t"}

# A small confusables table covering the Cyrillic/Greek letters most used to dodge filters.
_HOMOGLYPHS = str.maketrans(
    {
        "а": "a",
        "е": "e",
        "о": "o",
        "р": "p",
        "с": "c",
        "у": "y",
        "х": "x",
        "і": "i",
        "ј": "j",
        "ѕ": "s",
        "ԁ": "d",
        "ɡ": "g",
        "һ": "h",
        "ӏ": "l",
        "ո": "n",
        "ս": "u",
        "Α": "a",
        "Β": "b",
        "Ε": "e",
        "Ι": "i",
        "Κ": "k",
        "Μ": "m",
        "Ν": "n",
        "Ο": "o",
        "Ρ": "p",
        "Τ": "t",
        "Χ": "x",
        "Υ": "y",
        "Ζ": "z",
        "ο": "o",
        "ι": "i",
        "ν": "v",
        "А": "a",
        "В": "b",
        "Е": "e",
        "К": "k",
        "М": "m",
        "Н": "h",
        "О": "o",
        "Р": "p",
        "С": "c",
        "Т": "t",
        "Х": "x",
    }
)

_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})

_LEET_WORD = re.compile(r"\b\w*[a-z]\w*\b")
_B64_TOKEN = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")
_WS = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class NormalizedText:
    original_length: int
    text: str
    plain: str
    skeleton: str
    decoded_segments: tuple[str, ...] = ()
    removed_invisible: int = 0
    flags: tuple[str, ...] = field(default=())


def _strip_invisible(text: str) -> tuple[str, int]:
    out: list[str] = []
    removed = 0
    for ch in text:
        cat = unicodedata.category(ch)
        if ch in _KEEP_CONTROL or (cat not in {"Cf", "Cc", "Co", "Cs"}):
            out.append(ch)
        else:
            removed += 1
    return "".join(out), removed


def _deleet(text: str) -> str:
    # Only translate digits/symbols inside tokens that also contain letters ("1gn0re"), so plain
    # numbers ("call 555 0101") stay intact.
    return _LEET_WORD.sub(lambda m: m.group(0).translate(_LEET), text)


def _strip_marks(text: str) -> str:
    # Drop combining marks ("Zalgo" text, accents) so "ìgnórè" matches "ignore".
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


def make_plain(text: str) -> str:
    """Lowercased, homoglyph-mapped, accent-free view (no leetspeak mapping)."""
    s = _strip_marks(text.translate(_HOMOGLYPHS)).casefold().replace("’", "'").replace("‘", "'")
    return _WS.sub(" ", s).strip()


def make_skeleton(text: str) -> str:
    s = _strip_marks(text.translate(_HOMOGLYPHS)).casefold()
    s = _deleet(s.replace("@", "a").replace("$", "s"))
    # Join letters split by separators: "i.g.n.o.r.e" / "i-g-n-o-r-e" / "i g n o r e".
    s = re.sub(r"\b(?:[a-z][\s._\-*]){3,}[a-z]\b", lambda m: re.sub(r"[\s._\-*]", "", m.group(0)), s)
    s = s.replace("’", "'").replace("‘", "'")
    return _WS.sub(" ", s).strip()


def decode_base64_segments(text: str, require_words: bool = True) -> tuple[str, ...]:
    """Decode base64-looking tokens. ``require_words`` keeps only natural-language payloads."""
    decoded: list[str] = []
    for token in _B64_TOKEN.findall(text):
        padded = token + "=" * (-len(token) % 4)
        try:
            raw = base64.b64decode(padded, validate=True)
            candidate = raw.decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        printable = sum(ch.isprintable() or ch.isspace() for ch in candidate)
        if not candidate or printable / len(candidate) <= 0.95:
            continue
        if not require_words or re.search(r"[a-zA-Z]{3,}\s", candidate):
            decoded.append(candidate)
    return tuple(decoded)


def normalize(raw: str) -> NormalizedText:
    nfkc = unicodedata.normalize("NFKC", raw)
    text, removed = _strip_invisible(nfkc)
    text = text.strip()
    decoded = decode_base64_segments(text)
    flags: list[str] = []
    if removed:
        flags.append("invisible_characters_removed")
    if decoded:
        flags.append("base64_payload_decoded")
    if text.translate(_HOMOGLYPHS) != text:
        flags.append("homoglyphs_present")
    return NormalizedText(
        original_length=len(raw),
        text=text,
        plain=make_plain(text),
        skeleton=make_skeleton(text),
        decoded_segments=decoded,
        removed_invisible=removed,
        flags=tuple(flags),
    )
