"""Species/population normalization for Oral Bioavailability v3."""

from __future__ import annotations

import re
from typing import Any

NORMALIZED_COLUMN = "species_or_population_normalized"

NULL_VALUES = {
    "",
    "n/a",
    "na",
    "none",
    "not applicable",
    "not reported",
    "not specified",
    "not stated",
    "null",
    "unknown",
    "unspecified",
}

SPECIES_PATTERNS: dict[str, list[str]] = {
    "human": [
        r"\bhuman(s)?\b",
        r"\bpatient(s)?\b",
        r"\bvolunteer(s)?\b",
        r"\bsubject(s)?\b",
        r"\bparticipant(s)?\b",
        r"\bindividual(s)?\b",
        r"\bpeople\b",
        r"\bperson(s)?\b",
        r"\bmen\b",
        r"\bman\b",
        r"\bwomen\b",
        r"\bwoman\b",
        r"\bchild(ren)?\b",
        r"\bpediatric\b",
        r"\bpaediatric(s)?\b",
        r"\bpaediatrics\b",
        r"\bneonate(s)?\b",
        r"\bnewborn(s)?\b",
        r"\binfant(s)?\b",
        r"\belderly\b",
        r"\brecipient(s)?\b",
        r"^controls?$",
        r"^healthy control(s)?$",
        r"^control group$",
        r"^healthy population$",
        r"^general population$",
        r"^adult population$",
        r"^population average$",
        r"^young adult(s)?$",
        r"^younger adult(s)?$",
        r"^older adult(s)?$",
        r"^non[- ]?pregnant adult(s)?$",
        r"^non[- ]?pregnant population$",
        r"^non[- ]?pregnant state$",
        r"^healthy adult male(s)?$",
        r"^healthy male adult(s)?$",
        r"^healthy young male(s)?$",
        r"^young healthy male(s)?$",
        r"^young male$",
        r"^young female$",
        r"^males?$",
        r"^females?$",
        r"^normals?$",
        r"^\d+ normals?$",
        r"^\d+ adult(s)?$",
        r"^\d+ healthy adult male(s)?$",
        r"^normal adult(s)?$",
        r"^hospitalized male adult(s)?$",
        r"^healthy male adult(s)?$",
        r"^eight healthy females$",
        r"^males and females \(n=\d+\)$",
        r"^healthy japanese male(s)?$",
        r"^japanese$",
        r"^asian(s)?$",
        r"^non[- ]asian$",
        r"^chinese$",
        r"^humanos$",
        r"^adultos?$",
        r"^adulte$",
        r"^enfant$",
        r"^voluntarios$",
        r"^probanden$",
        r"^hv$",
        r"^adult(s)?$",
        r"^8 adult(s)?$",
        r"^healthy adult(s)?$",
        r"^healthy males?$",
        r"^healthy females?$",
        r"^normal controls?$",
        r"\bcaucasian(s)?\b",
    ],
    "rodent": [
        r"\brodent(s)?\b",
        r"\brat(s)?\b",
        r"\bwistar\b",
        r"\bsprague[-– ]?dawley\b",
        r"\bsd rat(s)?\b",
        r"\bf344\b",
        r"\bhan wistar\b",
        r"\bmouse\b",
        r"\bmice\b",
        r"\bmurine\b",
        r"\bcd[- ]?1\b",
        r"\bicr\b",
        r"\bbalb/?c\b",
        r"\bc57bl/?6j?\b",
        r"\bnude mice\b",
        r"\bhamster(s)?\b",
        r"\bguinea pig(s)?\b",
        r"\bwoodchuck(s)?\b",
    ],
    "dog": [
        r"\bdog(s)?\b",
        r"\bbeagle(s)?\b",
        r"\bcanine\b",
        r"\bcanine(s)?\b",
        r"\bgreyhound(s)?\b",
        r"\bhunden\b",
        r"\bhunde\b",
    ],
    "monkey": [
        r"\bmonkey(s)?\b",
        r"\bcynomolgus\b",
        r"\brhesus\b",
        r"\bmacaque(s)?\b",
        r"\bprimate(s)?\b",
        r"\bnon[- ]human primate(s)?\b",
        r"\bchimpanzee(s)?\b",
        r"\bcyno\b",
        r"\bmarmoset(s)?\b",
        r"\bbaboon(s)?\b",
        r"\bnhp\b",
        r"\bvervet monkey(s)?\b",
    ],
    "rabbit": [r"\brabbit(s)?\b"],
    "pig": [
        r"\bpig(s)?\b",
        r"\bswine\b",
        r"\bporcine\b",
        r"\bminipig(s)?\b",
        r"\bmini[- ]?pig(s)?\b",
        r"\bmicrominipig(s)?\b",
        r"\bpiglet(s)?\b",
    ],
    "horse": [
        r"\bhorse(s)?\b",
        r"\bfoal(s)?\b",
        r"\bpony|ponies\b",
        r"\bequine\b",
        r"\bdonkey(s)?\b",
    ],
    "cat": [
        r"\bcat(s)?\b",
        r"\bfeline\b",
    ],
    "bird": [
        r"\bbird(s)?\b",
        r"\bchicken(s)?\b",
        r"\bbroiler(s)?\b",
        r"\bturkey(s)?\b",
        r"\bduck(s)?\b",
        r"\bpigeon(s)?\b",
        r"\bgoose|geese\b",
        r"\bhen(s)?\b",
        r"\bchick(s)?\b",
        r"\bquail(s)?\b",
        r"\bcockatiel(s)?\b",
        r"\bmallard(s)?\b",
        r"\bparrot(s)?\b",
        r"\bwattlebird(s)?\b",
        r"\bostrich(es)?\b",
        r"\bpoultry\b",
    ],
    "ruminant": [
        r"\bsheep\b",
        r"\bewe(s)?\b",
        r"\bgoat(s)?\b",
        r"\blamb(s)?\b",
        r"\bcalf|calves\b",
        r"\bcattle\b",
        r"\bcow(s)?\b",
        r"\bbovine\b",
        r"\balpaca(s)?\b",
        r"\bllama(s)?\b",
    ],
    "fish": [
        r"\bfish\b",
        r"\bsalmon\b",
        r"\btrout\b",
        r"\bcatfish\b",
        r"\bflounder\b",
        r"\bturbot\b",
        r"\bcarp\b",
        r"\bcod\b",
        r"\btilapia\b",
        r"\bhalibut\b",
        r"\bkingfish\b",
        r"\beel(s)?\b",
        r"\bsea bass\b",
        r"\bsturgeon\b",
    ],
    "ferret": [r"\bferret(s)?\b"],
    "camelid": [
        r"\bcamel(s)?\b",
        r"\bcamelid(s)?\b",
    ],
    "marsupial": [
        r"\bkoala(s)?\b",
        r"\bpossum(s)?\b",
    ],
    "bat": [
        r"\bbat(s)?\b",
    ],
}


def normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return None if text in NULL_VALUES else text


def normalized_species_or_population(value: Any) -> str | None:
    text = normalize_text(value)
    if text is None:
        return None
    if text in {"animal", "animals", "animal models", "preclinical species"}:
        return None
    if re.search(r"\bguinea pig(s)?\b", text):
        return "rodent"
    if re.search(r"\bnon[- ]human primate(s)?\b", text):
        return "monkey"
    matches: set[str] = set()
    for label, patterns in SPECIES_PATTERNS.items():
        if any(re.search(pattern, text) for pattern in patterns):
            matches.add(label)
    if len(matches) == 1:
        return next(iter(matches))
    return None
