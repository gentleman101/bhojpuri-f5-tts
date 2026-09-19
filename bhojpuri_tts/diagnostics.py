"""Pronunciation contrasts where Bhojpuri is known to diverge from Hindi for the same spelling.

The model reads characters, so these are the places a Hindi-trained model is most likely to be wrong.
Each regex selects held-out sentences that exercise one contrast.
"""

CATEGORIES = {
    "v_to_b": (r"\bव[िीकर्]", "व often spoken as ब (विकास -> बिकास)"),
    "sibilant": (r"[शष]", "श/ष often merge to स"),
    "retroflex_n": (r"ण", "ण often spoken as न"),
    "avagraha": (r"ऽ", "ऽ lengthens the preceding vowel (बढ़ऽ)"),
    "verb_la": (r"(ला|ले|लस)\b", "Bhojpuri verb endings -ला/-ले"),
    "verb_ba": (r"\b(बा|बाड़|हवे|हव|हऽ)\b", "Bhojpuri copula बा/हवे vs Hindi है"),
    "final_vowel": (r"[क-ह]ल\b", "final vowel retained where Hindi drops it (कहल)"),
    "conjunct": (r"[क-ह]्[क-ह]्[क-ह]", "consonant clusters"),
}
