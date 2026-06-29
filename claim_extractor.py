"""
Claim extraction from backstory text using NLP with claim categorization
"""

import logging
import re
from typing import List, Dict, Optional
from nvidia_client import NVIDIAClient
from config import nlp

logger = logging.getLogger(__name__)

# Claim categories
CATEGORY_IDENTITY = "IDENTITY"
CATEGORY_SPATIAL = "SPATIAL"
CATEGORY_TEMPORAL = "TEMPORAL"
CATEGORY_RELATIONAL = "RELATIONAL"
CATEGORY_CAUSAL = "CAUSAL"

# Entity type -> claim category mapping for auto-categorisation
_ENT_TYPE_TO_CATEGORY: Dict[str, str] = {
    "PERSON": CATEGORY_IDENTITY,
    "ORG": CATEGORY_IDENTITY,
    "GPE": CATEGORY_SPATIAL,
    "LOC": CATEGORY_SPATIAL,
    "FAC": CATEGORY_SPATIAL,
    "DATE": CATEGORY_TEMPORAL,
    "TIME": CATEGORY_TEMPORAL,
    "CARDINAL": CATEGORY_TEMPORAL,
    "ORDINAL": CATEGORY_TEMPORAL,
}

# Keyword patterns for rule-based category detection
_RELATIONAL_KEYWORDS = re.compile(
    r"\b(father|mother|brother|sister|son|daughter|wife|husband|friend|"
    r"uncle|aunt|nephew|niece|cousin|spouse|partner|companion|mentor|"
    r"married|engaged|betrothed|related|family|kin|parent)\b",
    re.IGNORECASE,
)
_CAUSAL_KEYWORDS = re.compile(
    r"\b(because|caused|led to|resulted in|due to|therefore|hence|"
    r"consequently|killed|destroyed|built|created|decided|fled|escaped|"
    r"attacked|saved|rescued|discovered|found|lost|stole|bought|sold)\b",
    re.IGNORECASE,
)


def _auto_categorise(claim_text: str) -> str:
    """Assign a category to a claim based on its NER and keyword content."""
    doc = nlp(claim_text)

    # Check entity types
    ent_labels = {ent.label_ for ent in doc.ents}
    for label in ("GPE", "LOC", "FAC"):
        if label in ent_labels:
            return CATEGORY_SPATIAL
    for label in ("DATE", "TIME"):
        if label in ent_labels:
            return CATEGORY_TEMPORAL

    # Keyword-based checks (before PERSON check so relational/causal take priority)
    if _RELATIONAL_KEYWORDS.search(claim_text):
        return CATEGORY_RELATIONAL
    if _CAUSAL_KEYWORDS.search(claim_text):
        return CATEGORY_CAUSAL

    # PERSON entities suggest identity claims
    if "PERSON" in ent_labels:
        return CATEGORY_IDENTITY

    return CATEGORY_IDENTITY  # default


class ClaimExtractor:
    """Extract atomic factual claims from backstory text using NLP"""

    def __init__(self, llm: NVIDIAClient):
        self.llm = llm
        # Stores claim_text -> category for the last extraction.
        # Allows callers to look up categories without changing the return type.
        self.last_claim_categories: Dict[str, str] = {}
        logger.info("ClaimExtractor initialized")

    def extract_claims(self, backstory: str) -> List[str]:
        """
        Extract atomic factual claims from backstory text.
        Uses NLP to identify key entities and relations before LLM extraction.

        Returns:
            List of claim strings (backward compatible).
            Categories are stored in self.last_claim_categories as {claim_text: category}.
        """
        if not backstory or not backstory.strip():
            logger.warning("Empty backstory provided to extract_claims")
            self.last_claim_categories = {}
            return []

        # Pre-process with NLP to identify key entities
        doc = nlp(backstory)

        persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
        locations = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]

        entity_context = ""
        if persons:
            entity_context += f"\nKey characters: {', '.join(set(persons[:5]))}"
        if dates:
            entity_context += f"\nKey dates: {', '.join(set(dates[:5]))}"
        if locations:
            entity_context += f"\nKey locations: {', '.join(set(locations[:5]))}"

        prompt = f"""### ROLE
You are a Narrative Logic Auditor. Your task is to decompose a character backstory into unique, atomic, and verifiable "Narrative Anchors" to be checked against a source text.

### INPUT
BACKSTORY:
{backstory}
{entity_context}

### EXTRACTION RULES (STRICT)
1. **Atomic Only**: Each claim must be a single, independent fact. No "and," "but," or "because."
2. **De-duplication**: Do not extract the same fact twice, even if phrased differently in the text.
3. **Focus on Anchors**: Prioritize "Hard Facts" that are likely to be contradicted if the backstory is fake:
    - **IDENTITY**: Names, roles, professions, species, titles, physical traits.
    - **SPATIAL**: Specific locations (cities, rooms, countries).
    - **TEMPORAL**: Specific times (years, ages, durations, sequences).
    - **RELATIONAL**: Family ties, friendships, specific interactions with other characters.
    - **CAUSAL**: Specific events, actions, cause-and-effect chains.
4. **Discard Fluff**: Ignore subjective descriptions (e.g., "He was very brave") unless they are core character traits mentioned in the narrative.
5. **Entity-Focused**: Ensure claims reference the key entities identified above.

### OUTPUT FORMAT
Return ONLY a newline-separated list of claims. Each claim MUST start with a category tag in brackets, followed by the claim text.
- No numbering.
- No introductory text.

### EXAMPLES

Example 1 (Consistent backstory):
Input: "In 1995, Elara moved to London after her father, a clockmaker, died in a fire."
Output:
[TEMPORAL] Elara moved to London in 1995
[IDENTITY] Elara's father was a clockmaker
[CAUSAL] Elara's father died in a fire
[CAUSAL] Elara moved to London after her father's death

Example 2 (Potentially inconsistent backstory):
Input: "Captain Ahab lost his leg to Moby Dick during a voyage on the Pequod in the Atlantic Ocean."
Output:
[IDENTITY] Ahab held the rank of Captain
[CAUSAL] Ahab lost his leg to Moby Dick
[SPATIAL] The voyage took place on the Pequod
[SPATIAL] The voyage occurred in the Atlantic Ocean

Example 3 (Relational claims):
Input: "Harry was raised by his aunt Petunia and uncle Vernon after his parents were killed by Voldemort."
Output:
[RELATIONAL] Petunia is Harry's aunt
[RELATIONAL] Vernon is Harry's uncle
[CAUSAL] Harry was raised by Petunia and Vernon
[CAUSAL] Harry's parents were killed by Voldemort
[CAUSAL] Harry was raised by his aunt and uncle because his parents died

Claims:"""

        try:
            response = self.llm.chat(prompt)

            # Check if response is a JSON error from stub
            if response.strip().startswith("{") and '"error"' in response:
                logger.warning(
                    "Stub fallback detected in claim extraction. "
                    "Using spaCy-based claim extraction."
                )
                claims = self._extract_claims_spacy(backstory, persons, dates, locations)
                return claims

            # Parse response: split by newlines and clean
            raw_claims: List[str] = []
            categories: Dict[str, str] = {}

            for line in response.split("\n"):
                line = line.strip()
                if not line or line.startswith(("#", "-", "*", "{", "}")):
                    continue

                # Try to extract category tag  [CATEGORY] claim text
                cat_match = re.match(r"\[(\w+)\]\s*(.*)", line)
                if cat_match:
                    cat = cat_match.group(1).upper()
                    claim_text = cat_match.group(2).strip()
                    if cat not in (
                        CATEGORY_IDENTITY,
                        CATEGORY_SPATIAL,
                        CATEGORY_TEMPORAL,
                        CATEGORY_RELATIONAL,
                        CATEGORY_CAUSAL,
                    ):
                        cat = _auto_categorise(claim_text)
                else:
                    claim_text = line
                    cat = _auto_categorise(line)

                if claim_text and self._validate_claim(claim_text, persons, dates, locations):
                    raw_claims.append(claim_text)
                    categories[claim_text] = cat

            self.last_claim_categories = categories
            logger.info(f"Extracted {len(raw_claims)} validated claims from backstory")
            return raw_claims

        except Exception as e:
            logger.error(f"Error extracting claims: {e}")
            try:
                claims = self._extract_claims_spacy(backstory, persons, dates, locations)
                return claims
            except Exception:
                self.last_claim_categories = {}
                return []

    # ------------------------------------------------------------------ #
    # Improved spaCy-based fallback
    # ------------------------------------------------------------------ #

    def _extract_claims_spacy(
        self,
        backstory: str,
        persons: List[str],
        dates: List[str],
        locations: List[str],
    ) -> List[str]:
        """
        When LLM is unavailable, extract claims using spaCy dependency
        parsing.  Focus on sentences with named entities and extract
        subject-verb-object triples where possible.
        """
        doc = nlp(backstory)
        claims: List[str] = []
        categories: Dict[str, str] = {}

        for sent in doc.sents:
            sent_text = sent.text.strip()
            if len(sent_text.split()) < 4:
                continue

            # Check if sentence has named entities (prioritise these)
            sent_ents = [ent for ent in sent.ents]
            has_entities = len(sent_ents) > 0

            if has_entities:
                # Try SVO extraction
                svo_claims = self._extract_svo(sent)
                if svo_claims:
                    for c in svo_claims:
                        cat = self._categorise_from_entities(sent_ents)
                        claims.append(c)
                        categories[c] = cat
                else:
                    # Fall back to full sentence as claim
                    cat = self._categorise_from_entities(sent_ents)
                    claims.append(sent_text)
                    categories[sent_text] = cat
            elif len(sent_text.split()) > 6:
                # No entities but substantive sentence
                claims.append(sent_text)
                categories[sent_text] = _auto_categorise(sent_text)

        # Limit and store
        claims = claims[:12]
        self.last_claim_categories = {c: categories.get(c, CATEGORY_IDENTITY) for c in claims}
        logger.info(f"Fallback extraction: {len(claims)} claims via spaCy")
        return claims

    @staticmethod
    def _extract_svo(sent) -> List[str]:
        """Extract subject-verb-object triples from a spaCy Span."""
        triples: List[str] = []
        for token in sent:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                # Find subject
                subjects = [
                    child
                    for child in token.children
                    if child.dep_ in ("nsubj", "nsubjpass")
                ]
                # Find object
                objects = [
                    child
                    for child in token.children
                    if child.dep_ in ("dobj", "attr", "prep", "oprd")
                ]
                for subj in subjects:
                    subj_text = " ".join(
                        t.text for t in subj.subtree
                    )
                    verb_text = token.lemma_
                    for obj in objects:
                        obj_text = " ".join(
                            t.text for t in obj.subtree
                        )
                        triple = f"{subj_text} {verb_text} {obj_text}"
                        if len(triple.split()) >= 3:
                            triples.append(triple)
                    if not objects and subjects:
                        # Intransitive verb – just subj + verb
                        triple = f"{subj_text} {verb_text}"
                        if len(triple.split()) >= 3:
                            triples.append(triple)
        return triples

    @staticmethod
    def _categorise_from_entities(ents) -> str:
        """Pick category from a list of spaCy entities."""
        labels = {ent.label_ for ent in ents}
        for label in ("GPE", "LOC", "FAC"):
            if label in labels:
                return CATEGORY_SPATIAL
        for label in ("DATE", "TIME"):
            if label in labels:
                return CATEGORY_TEMPORAL
        if labels & {"PERSON", "ORG"}:
            if len(labels & {"PERSON", "ORG"}) >= 2:
                return CATEGORY_RELATIONAL
            return CATEGORY_IDENTITY
        return CATEGORY_IDENTITY

    # ------------------------------------------------------------------ #

    def _validate_claim(
        self,
        claim: str,
        persons: List[str],
        dates: List[str],
        locations: List[str],
    ) -> bool:
        """
        Validate that claim contains at least one key entity or is substantive.
        """
        claim_lower = claim.lower()

        for person in persons:
            if person.lower() in claim_lower:
                return True
        for date in dates:
            if date.lower() in claim_lower:
                return True
        for location in locations:
            if location.lower() in claim_lower:
                return True

        # If no entities, check if claim is substantive (>5 words)
        return len(claim.split()) > 5
