"""
Stage T017: Syntactic Profiler (Syntactic Fingerprinting)
Analyzes how language is constructed in code-mixed Romanized Hinglish.
Quantifies sentence complexity, clause structure (verbless fragments, simple,
compound, complex), speech acts (imperatives, inquisitives, declarations),
negation mechanics, conditionals, pronoun orientation, and discourse particles.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Coordinating conjunctions (compound sentence markers)
COORDINATING_CONJUNCTIONS: Set[str] = {
    "and", "aur", "but", "lekin", "par", "ya", "or", "waise", "plus",
}

# Subordinating conjunctions (complex sentence markers)
SUBORDINATING_CONJUNCTIONS: Set[str] = {
    "kyuki", "kyunki", "because", "agar", "if", "jab", "jabki", "tab",
    "since", "although", "though", "unless", "while", "whereas", "warna",
    "jabse", "jaha", "jahan", "jaise", "jaisa", "jaisi",
}

# Question / Interrogative words
INTERROGATIVE_TOKENS: Set[str] = {
    "kya", "kyu", "kyun", "kab", "kaha", "kahan", "kaise", "kese",
    "kon", "kaun", "kiska", "kisko", "kisse", "kidhar", "kitna",
    "kitne", "kitni", "what", "why", "when", "where", "how", "who",
    "whom", "whose", "which",
}

# Imperative verb roots (commands / directives common in Hinglish & English)
IMPERATIVE_VERB_ROOTS: Set[str] = {
    "dekh", "dekho", "bata", "btao", "bol", "bolo", "bhej", "bhejo",
    "padh", "soja", "so", "kar", "karo", "check", "run", "pull", "push",
    "text", "call", "chal", "chalo", "aaja", "aa", "ruk", "ruko",
    "sun", "suno", "laga", "daal", "chod", "chodd", "le", "de",
    "send", "try", "test", "fix", "open", "share", "ask",
}

# Negation particles
NEGATION_TOKENS: Set[str] = {
    "nahi", "nhi", "na", "mat", "no", "not", "never", "neither", "nor", "none",
}

# Pronoun categories
PRONOUNS_1ST_PERSON: Set[str] = {
    "me", "mein", "mai", "mene", "maine", "mera", "meri", "mere",
    "mujhe", "mujhko", "i", "my", "mine", "we", "our", "ours", "us",
    "hum", "hume", "humko", "apna", "apne", "apni",
}

PRONOUNS_2ND_PERSON: Set[str] = {
    "tu", "tune", "tera", "teri", "tere", "tujhe", "tujhko",
    "tum", "tumhe", "tumhara", "tumhari", "tumhare", "aap", "aapka",
    "you", "your", "yours", "u", "ur",
}

PRONOUNS_3RD_PERSON: Set[str] = {
    "wo", "woh", "uske", "uski", "uska", "unka", "unki", "unke",
    "usse", "unhe", "unko", "ye", "yeh", "inhe", "inka", "iski", "iske",
    "he", "him", "his", "she", "her", "hers", "they", "them", "their",
    "theirs", "it", "its",
}

# Modals and obligation markers
MODAL_TOKENS: Set[str] = {
    "chaiye", "chahiye", "sakta", "sakte", "sakti", "padega", "padegi",
    "padenge", "hoga", "hogi", "honge", "can", "could", "should", "would",
    "must", "might", "may", "need", "have_to",
}

# Conversational discourse enclitics / slang particles
DISCOURSE_ENCLITICS: Set[str] = {
    "re", "be", "yaar", "bhai", "na", "bhi", "toh", "bc", "bkl", "bhenchod",
}

RE_WORD_TOKENS = re.compile(r"[a-zA-Z\u0900-\u097F0-9_']+")


@dataclass
class SyntacticProfile:
    """Syntactic distribution profile for a corpus or communication archetype."""

    total_turns: int

    # Clause Structure
    clause_structure: Dict[str, float]

    # Speech Acts & Sentence Functions
    speech_acts: Dict[str, float]

    # Negation & Conditionals
    negation_and_conditionals: Dict[str, float]

    # Pronoun Orientation
    pronoun_orientation: Dict[str, float]

    # Modality & Discourse Enclitics
    modality_and_discourse: Dict[str, Any]

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_turns": self.total_turns,
            "clause_structure": self.clause_structure,
            "speech_acts": self.speech_acts,
            "negation_and_conditionals": self.negation_and_conditionals,
            "pronoun_orientation": self.pronoun_orientation,
            "modality_and_discourse": self.modality_and_discourse,
            "metadata": self.metadata,
        }


class SyntacticProfiler:
    """
    Analyzes syntactic constructions, clause structures, speech acts,
    and conversational mechanics in code-mixed Hinglish.
    """

    def analyze_turn_syntax(self, turn_text: str) -> Dict[str, Any]:
        """
        Extracts syntactic markers from a single conversational turn.
        """
        raw_text = turn_text.strip()
        tokens = [t.lower() for t in RE_WORD_TOKENS.findall(raw_text)]
        word_count = len(tokens)
        token_set = set(tokens)

        # 1. Question / Interrogative Detection
        has_question_mark = "?" in raw_text
        has_wh_token = any(t in INTERROGATIVE_TOKENS for t in tokens)
        is_inquisitive = has_question_mark or has_wh_token

        # 2. Negation Detection
        has_negation = any(t in NEGATION_TOKENS for t in tokens)
        negation_count = sum(1 for t in tokens if t in NEGATION_TOKENS)

        # 3. Conditional Detection (e.g. agar...toh, if...then, jab...tab)
        has_conditional = (
            ("agar" in token_set or "if" in token_set)
            or ("jab" in token_set and "tab" in token_set)
            or ("warna" in token_set)
        )

        # 4. Imperative / Directive Detection
        # Check if first non-empty word is an imperative verb, or starts with "mat <verb>"
        is_imperative = False
        if tokens:
            first_word = tokens[0]
            if first_word in IMPERATIVE_VERB_ROOTS:
                is_imperative = True
            elif len(tokens) >= 2 and first_word in {"mat", "don't", "dont"} and tokens[1] in IMPERATIVE_VERB_ROOTS:
                is_imperative = True
            elif len(tokens) >= 2 and tokens[1] in {"kar", "karo", "dekh", "bata"}:
                is_imperative = True

        # 5. Clause Structure & Complexity
        # Verbless Fragment: sub-3 words, pure emoji/slang, or short confirmation
        has_coord = any(t in COORDINATING_CONJUNCTIONS for t in tokens)
        has_subord = any(t in SUBORDINATING_CONJUNCTIONS for t in tokens)

        if word_count <= 2:
            clause_type = "verbless_fragment"
        elif has_subord:
            clause_type = "complex_subordinate"
        elif has_coord:
            clause_type = "compound_coordinate"
        else:
            clause_type = "simple_clause"

        # 6. Pronoun Orientation
        has_1st = any(t in PRONOUNS_1ST_PERSON for t in tokens)
        has_2nd = any(t in PRONOUNS_2ND_PERSON for t in tokens)
        has_3rd = any(t in PRONOUNS_3RD_PERSON for t in tokens)

        # 7. Modality
        has_modal = any(t in MODAL_TOKENS for t in tokens)

        # 8. Discourse Enclitics
        enclitics = [t for t in tokens if t in DISCOURSE_ENCLITICS]

        return {
            "word_count": word_count,
            "tokens": tokens,
            "clause_type": clause_type,
            "is_imperative": is_imperative,
            "is_inquisitive": is_inquisitive,
            "has_question_mark": has_question_mark,
            "has_wh_token": has_wh_token,
            "has_negation": has_negation,
            "negation_count": negation_count,
            "has_conditional": has_conditional,
            "has_1st_person": has_1st,
            "has_2nd_person": has_2nd,
            "has_3rd_person": has_3rd,
            "has_modal": has_modal,
            "enclitics": enclitics,
        }

    def fit(self, pairs: Sequence[Dict[str, Any]], name: str = "global") -> SyntacticProfile:
        """
        Fits the Syntactic Profiler across all pairs to compute empirical distributions.
        """
        n_turns = len(pairs)
        if n_turns == 0:
            raise ValueError("Pairs sequence cannot be empty for syntactic profiling.")

        clause_counter: Counter[str] = Counter()
        imperative_count = 0
        inquisitive_count = 0
        negation_count = 0
        conditional_count = 0

        first_person_count = 0
        second_person_count = 0
        third_person_count = 0

        modal_count = 0
        enclitic_counter: Counter[str] = Counter()

        for p in pairs:
            raw_text = p.get("target_text", "")
            feats = self.analyze_turn_syntax(raw_text)

            clause_counter[feats["clause_type"]] += 1

            if feats["is_imperative"]:
                imperative_count += 1
            if feats["is_inquisitive"]:
                inquisitive_count += 1
            if feats["has_negation"]:
                negation_count += 1
            if feats["has_conditional"]:
                conditional_count += 1

            if feats["has_1st_person"]:
                first_person_count += 1
            if feats["has_2nd_person"]:
                second_person_count += 1
            if feats["has_3rd_person"]:
                third_person_count += 1

            if feats["has_modal"]:
                modal_count += 1

            enclitic_counter.update(feats["enclitics"])

        # Clause structure ratios
        clause_dist = {
            "verbless_fragment_ratio": round(clause_counter["verbless_fragment"] / n_turns, 4),
            "simple_clause_ratio": round(clause_counter["simple_clause"] / n_turns, 4),
            "compound_coordinate_ratio": round(clause_counter["compound_coordinate"] / n_turns, 4),
            "complex_subordinate_ratio": round(clause_counter["complex_subordinate"] / n_turns, 4),
        }

        # Speech acts
        speech_acts = {
            "imperative_directive_ratio": round(imperative_count / n_turns, 4),
            "inquisitive_question_ratio": round(inquisitive_count / n_turns, 4),
            "declarative_ratio": round((n_turns - imperative_count - inquisitive_count) / n_turns, 4),
        }

        # Negation & Conditionals
        neg_cond = {
            "negation_turns_ratio": round(negation_count / n_turns, 4),
            "conditional_turns_ratio": round(conditional_count / n_turns, 4),
        }

        # Pronoun Orientation
        pronouns = {
            "first_person_ratio": round(first_person_count / n_turns, 4),
            "second_person_ratio": round(second_person_count / n_turns, 4),
            "third_person_ratio": round(third_person_count / n_turns, 4),
            "self_to_other_ratio": round(first_person_count / max(second_person_count, 1), 2),
        }

        # Modality & Discourse Enclitics
        modality = {
            "modal_turns_ratio": round(modal_count / n_turns, 4),
            "enclitic_turns_ratio": round(sum(enclitic_counter.values()) / n_turns, 4),
            "top_enclitics": [
                {"particle": p, "count": cnt, "per_turn_ratio": round(cnt / n_turns, 4)}
                for p, cnt in enclitic_counter.most_common(8)
            ],
        }

        return SyntacticProfile(
            total_turns=n_turns,
            clause_structure=clause_dist,
            speech_acts=speech_acts,
            negation_and_conditionals=neg_cond,
            pronoun_orientation=pronouns,
            modality_and_discourse=modality,
            metadata={"profile_name": name},
        )

    def analyze_by_style(
        self,
        pairs: Sequence[Dict[str, Any]],
        style_labels: np.ndarray,
        style_names: Optional[Dict[int, str]] = None,
    ) -> Dict[str, SyntacticProfile]:
        """
        Computes syntactic profiles broken down across communication archetypes.
        """
        if len(pairs) != len(style_labels):
            raise ValueError("Lengths of pairs and style_labels must match.")

        unique_styles = sorted(np.unique(style_labels))
        profiles: Dict[str, SyntacticProfile] = {}

        for s in unique_styles:
            s_mask = style_labels == s
            s_pairs = [pairs[i] for i in range(len(pairs)) if s_mask[i]]
            s_name = style_names.get(int(s), f"Style_{s:02d}") if style_names else f"Style_{s:02d}"
            profiles[s_name] = self.fit(s_pairs, name=s_name)

        return profiles
