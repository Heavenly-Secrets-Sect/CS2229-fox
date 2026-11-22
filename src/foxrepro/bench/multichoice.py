"""Answer extraction for the two multiple-choice benchmarks (Table 11).

MMBench and MathVista-MC both ask for an option LETTER. This module turns a
model's free text into a letter by RULE, with no LLM in the loop, and says
which rule fired so that a score can be audited for how much of it rests on
the weak rules.

The rule, in order; the first that fires wins:

1. ``exact``    the whole answer, stripped of whitespace, quotes, brackets and
                a trailing period, is one valid option letter ("B", "(b).").
2. ``lead``     the answer STARTS with an uppercase valid letter followed by
                a delimiter or the end: "B. a cat", "(C) 97", "A: ...".
                A bare leading word such as "A dog is shown" does NOT match,
                because a space is not a delimiter -- otherwise the article
                "A" would be read as option A.
3. ``phrase``   the LAST "answer is X" / "option X" / "choice X" / "answer:
                X" with X a valid letter. Last, because a derivation that
                states an intermediate "option A" and concludes "the answer
                is C" means C.
4. ``tail``     (MathVista only, whose prompt asks for the letter "at the
                end") a valid uppercase letter, optionally bracketed, as the
                last token of the answer.
5. ``text``     exactly one option's TEXT is the whole answer, or appears in
                it delimited by non-alphanumerics, and no other option's text
                does. Ambiguity is not guessed through.

Anything else is UNPARSED. An unparsed answer is a WRONG answer and is
counted and reported, never dropped: dropping would let a model raise its
score by refusing to answer. A letter that is not one of the question's
options (say "D" on a three-option question) is also unparsed, and is counted
separately as ``invalid_letter`` so it can be told from rambling.

The official MathVista protocol uses an LLM to extract the answer; this is
the declared rule-based substitute (docs/DEVIATIONS.md). MMBench's own
pipeline falls back to an LLM too; the same substitution applies.
"""
from __future__ import annotations

import re
from typing import Dict, Optional, Sequence, Tuple

__all__ = ["extract_choice", "RULES"]

RULES: Tuple[str, ...] = ("exact", "lead", "phrase", "tail", "text")

_STRIP = " \t\r\n\"'`*()[]{}."
_LEAD = re.compile(r"^[\(\[]?([A-Z])(?:[\.\)\]:,]|$)")
# The letter must be UPPERCASE and followed by a delimiter or the end:
# "the answer is a cat" and "the answer is A dog" are sentences, not letters.
_PHRASE = re.compile(
    r"(?i:answer|option|choice)\s*(?i:is|:)?\s*[\(\[]?([A-Z])"
    r"(?=[\)\]\.,:;]|\s*$|\s*\n)")
_TAIL = re.compile(r"(?:^|[\s\(\[])([A-Z])[\)\]]?[\.\s]*$")


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().strip(_STRIP)).lower()


def _contains_delimited(hay: str, needle: str) -> bool:
    if not needle:
        return False
    pat = r"(?<![0-9a-z])" + re.escape(needle) + r"(?![0-9a-z])"
    return re.search(pat, hay) is not None


def extract_choice(
    text: str,
    options: Dict[str, str],
    allow_tail: bool = False,
) -> Tuple[Optional[str], str]:
    """Extract an option letter from ``text``.

    Args:
        text: the model's output.
        options: ``{"A": "text", ...}`` -- the options THIS question has.
        allow_tail: enable rule 4 (MathVista).

    Returns:
        ``(letter, rule)``. ``letter`` is ``None`` when unparsed, and ``rule``
        is then ``"none"`` (nothing recognisable) or ``"invalid_letter"`` (a
        letter was given that this question does not offer).
    """
    valid = set(options)
    raw = (text or "").strip()
    if not raw:
        return None, "none"

    # 1. exact
    bare = raw.strip(_STRIP)
    if len(bare) == 1 and bare.isalpha():
        letter = bare.upper()
        return (letter, "exact") if letter in valid else (None, "invalid_letter")

    # 2. leading letter
    m = _LEAD.match(raw)
    if m:
        letter = m.group(1)
        if letter in valid:
            return letter, "lead"
        # "A" opening a sentence is far more often the article than a
        # missing option, so an out-of-range leading letter is only called
        # invalid when a delimiter followed it (the regex already requires
        # one), which is the case here.
        return None, "invalid_letter"

    # 3. last "answer is X"
    hits = _PHRASE.findall(raw)
    if hits:
        letter = hits[-1].upper()
        if letter in valid:
            return letter, "phrase"
        return None, "invalid_letter"

    # 4. trailing letter
    if allow_tail:
        m = _TAIL.search(raw)
        if m:
            letter = m.group(1)
            if letter in valid:
                return letter, "tail"
            return None, "invalid_letter"

    # 5. option text
    hay = _norm_text(raw)
    norm = {k: _norm_text(v) for k, v in options.items()}
    whole = [k for k, v in norm.items() if v and v == hay]
    if len(whole) == 1:
        return whole[0], "text"
    inside = [k for k, v in norm.items() if _contains_delimited(hay, v)]
    # A text nested inside another matched text ("2" inside "2.5") is the
    # same mention, not a second candidate: keep only the maximal ones.
    inside = [k for k in inside
              if not any(k != j and norm[k] != norm[j] and norm[k] in norm[j]
                         for j in inside)]
    if len(inside) == 1:
        return inside[0], "text"
    return None, "none"
