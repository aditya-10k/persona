"""
Stage T018: Code-Switching & Language Analyzer
Performs token-level language identification (English vs Romanized Hindi vs Universal),
context-aware homograph disambiguation, turn-level mode classification (pure English,
pure Hindi, code-switched Hinglish), switch-point frequency quantification,
and borrowed English loanword extraction.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Curated high-precision Romanized Hindi vocabulary
HINDI_LEXICON: Set[str] = {
    # Pronouns & determiners
    "me", "mein", "mai", "mene", "maine", "mera", "meri", "mere", "mujhe",
    "mujhko", "hum", "hume", "humko", "apna", "apne", "apni", "tu", "tune",
    "tera", "teri", "tere", "tujhe", "tujhko", "tum", "tumhe", "tumhara",
    "tumhari", "tumhare", "aap", "aapka", "aapke", "aapki", "wo", "woh",
    "uske", "uska", "uski", "unka", "unki", "unke", "usse", "unhe", "unko",
    "ye", "yeh", "inhe", "inka", "iski", "iske", "isko", "isse", "yehi", "wohi",
    "kisi", "koi", "sab", "sabko", "kuch",
    # Copulas & auxiliaries
    "hai", "hain", "tha", "thi", "the", "ho", "hua", "hue", "hui", "hu", "h",
    "raha", "rahi", "rahe", "rha", "rhi", "rhe", "padega", "padegi", "padenge",
    "hoga", "hogi", "honge", "honga", "hota", "hoti", "hote", "chaiye", "chahiye",
    "sakta", "sakte", "sakti",
    # Verb roots, nouns & inflections
    "kar", "karna", "karke", "karo", "kare", "karega", "karegi", "karenge",
    "kiya", "kia", "diya", "dia", "liya", "lia", "gaya", "gya", "gayi", "gaye", "gyi", "gye",
    "aaya", "aaye", "aayi", "aana", "chal", "chalo", "chalra", "chalte", "dekh",
    "dekho", "dekha", "dekhne", "dekhna", "bata", "btao", "bol", "bolo", "bola",
    "bolra", "bolna", "sun", "suno", "sunna", "aaja", "aa", "ja", "jaa", "jaana",
    "jana", "lega", "legi", "lenge", "dunga", "de", "dede", "lele", "lere", "dere",
    "lag", "laga", "lage", "lagra", "lagti", "lagta", "milte", "mil", "mila", "mili",
    "mile", "milna", "ruk", "ruko", "soch", "sochra", "socha", "sochna", "samjha",
    "samajh", "samajhna", "pata", "chodd", "chod", "chodna", "padh", "padhna",
    "soja", "bhej", "bhejo", "bheja", "bhejna", "rakh", "rakho", "rakhna", "laya",
    "lao", "dhundne", "dhund", "kaam", "kam", "baat", "baatein", "bat", "cheez",
    # Discourse markers, adverbs & particles
    "bhai", "yaar", "bro", "bc", "bkl", "nahi", "nhi", "na", "mat", "haan",
    "haa", "ha", "acha", "accha", "achha", "theek", "thik", "arre", "are", "abe",
    "abey", "aisa", "aise", "waisa", "waise", "thoda", "thode", "thodi", "bahut",
    "bohot", "abhi", "tab", "jab", "kab", "kyu", "kyun", "kya", "kaise", "kese",
    "kaha", "kahan", "kidhar", "kitna", "kitne", "kitni", "kon", "kaun", "yaha",
    "yha", "waha", "wha", "idhar", "udhar", "islie", "isliye", "isme", "usme",
    "kisme", "jisme", "toh", "to", "bhi", "hi", "he", "hena", "haina", "nhina",
    "bas", "fir", "phir", "pehle", "baad", "sath", "saath", "bina", "taraf",
    "pas", "paas", "samne", "wahi", "whi", "sahi", "galat", "pakka", "sach",
    "jhoot", "mast", "badhiya", "khatam", "shuru", "chalu", "band", "zyada",
    "jyada", "aur", "lekin", "par", "pe", "se", "ke", "ka", "ki", "ko", "ne",
    "ek", "do", "teen", "char", "ya", "re", "be", "clg", "grp", "oa", "sem", "yr",
    # Slang & colloquial
    "bhenchod", "bhadwe", "chutiya", "randi", "gand", "muth", "bakchodi",
    "nad", "laude", "bhide", "fati",
}

# Curated high-frequency English vocabulary
ENGLISH_LEXICON: Set[str] = {
    # Pronouns, determiners & articles
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "their", "our", "its", "mine", "yours", "theirs", "ours",
    "the", "a", "an", "this", "that", "these", "those", "what", "which", "who",
    "whom", "whose", "why", "how", "when", "where", "all", "any", "some", "both",
    "each", "every", "other", "another", "such", "no", "one", "two", "much", "many",
    "more", "most", "same", "something", "anything", "nothing", "everything",
    "someone", "anyone", "everyone", "nobody", "somebody",
    # Verbs & auxiliaries
    "is", "am", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "doing", "done", "will", "would", "shall", "should", "can", "could",
    "may", "might", "must", "go", "goes", "went", "gone", "going", "come", "came", "coming",
    "see", "saw", "seen", "seeing", "get", "got", "getting", "gotten", "make", "made", "making",
    "know", "knew", "known", "knowing", "take", "took", "taken", "taking", "give", "gave",
    "given", "giving", "find", "found", "finding", "think", "thought", "thinking", "tell",
    "told", "telling", "say", "said", "saying", "call", "called", "calling", "try", "tried",
    "trying", "need", "needed", "needing", "needs", "feel", "feels", "felt", "feeling",
    "become", "became", "becoming", "leave", "leaves", "left", "leaving", "put", "puts", "putting",
    "mean", "means", "meant", "meaning", "keep", "keeps", "kept", "keeping", "let", "lets",
    "begin", "began", "begun", "beginning", "seem", "seems", "seemed", "seeming", "help",
    "helps", "helped", "helping", "talk", "talks", "talked", "talking", "turn", "turns",
    "turned", "turning", "start", "starts", "started", "starting", "show", "shows", "showed",
    "showing", "hear", "hears", "heard", "hearing", "play", "plays", "played", "playing",
    "run", "runs", "ran", "running", "move", "moves", "moved", "moving", "like", "likes",
    "liked", "liking", "live", "lives", "lived", "living", "believe", "believed", "hold",
    "holds", "held", "holding", "bring", "brings", "brought", "bringing", "happen", "happens",
    "happened", "happening", "write", "writes", "wrote", "written", "writing", "provide",
    "provides", "provided", "sit", "sits", "sat", "sitting", "stand", "stands", "stood",
    "standing", "lose", "loses", "lost", "losing", "pay", "pays", "paid", "paying",
    "meet", "meets", "met", "meeting", "include", "includes", "included", "continue",
    "continues", "continued", "set", "sets", "setting", "learn", "learns", "learned",
    "learning", "change", "changes", "changed", "changing", "lead", "leads", "led",
    "leading", "understand", "understood", "watch", "watches", "watched", "watching",
    "follow", "follows", "followed", "following", "stop", "stops", "stopped", "stopping",
    "create", "creates", "created", "creating", "speak", "speaks", "spoke", "spoken",
    "speaking", "read", "reads", "reading", "allow", "allows", "allowed", "add", "adds",
    "added", "adding", "spend", "spends", "spent", "spending", "grow", "grows", "grew",
    "grown", "growing", "open", "opens", "opened", "opening", "walk", "walks", "walked",
    "walking", "win", "wins", "won", "winning", "offer", "offers", "offered", "remember",
    "remembers", "remembered", "love", "loves", "loved", "loving", "consider", "considered",
    "appear", "appears", "appeared", "buy", "buys", "bought", "buying", "wait", "waits",
    "waited", "waiting", "serve", "served", "die", "died", "send", "sends", "sent", "sending",
    "expect", "expects", "expected", "build", "builds", "built", "building", "stay", "stays",
    "stayed", "staying", "fall", "falls", "fell", "fallen", "cut", "cuts", "cutting",
    "reach", "reaches", "reached", "kill", "kills", "killed", "remain", "remains", "remained",
    "suggest", "suggests", "suggested", "raise", "raised", "pass", "passes", "passed", "passing",
    "sell", "sells", "sold", "selling", "require", "requires", "required", "report", "reports",
    "reported", "decide", "decides", "decided", "pull", "pulls", "pulled", "pulling",
    "push", "pushes", "pushed", "pushing", "test", "tests", "tested", "testing",
    "check", "checks", "checked", "checking", "fix", "fixes", "fixed", "fixing",
    "deploy", "deploys", "deployed", "deploying", "host", "hosts", "hosted", "hosting",
    "update", "updates", "updated", "updating", "install", "installs", "installed", "installing",
    "commit", "commits", "committed", "committing", "merge", "merges", "merged", "merging",
    "debug", "debugs", "debugged", "debugging", "login", "register", "apply", "applies",
    "applied", "applying", "type", "types", "typed", "typing", "work", "works", "worked",
    "working", "use", "uses", "used", "using", "want", "wants", "wanted", "wanting",
    "look", "looks", "looked", "looking", "ask", "asks", "asked", "asking", "post", "posts",
    "posted", "posting", "share", "shares", "shared", "sharing", "join", "joins", "joined",
    "joining", "pick", "picks", "picked", "picking", "drop", "drops", "dropped", "dropping",
    "connect", "connected", "connecting", "handle", "handled", "handling", "solve", "solved",
    "solving", "manage", "managed", "managing", "save", "saves", "saved", "saving",
    "load", "loads", "loaded", "loading", "fetch", "fetches", "fetched", "fetching",
    "parse", "parses", "parsed", "parsing", "render", "renders", "rendered", "rendering",
    "click", "clicks", "clicked", "clicking", "press", "select", "switch", "verify",
    "confirm", "cancel", "delete", "remove", "clear", "reset", "export", "import",
    # Prepositions & Conjunctions
    "of", "in", "to", "for", "with", "on", "at", "from", "by", "about", "as",
    "into", "like", "through", "after", "over", "between", "out", "against",
    "during", "without", "before", "under", "around", "among", "and", "but",
    "or", "nor", "so", "yet", "because", "although", "since", "unless",
    # Adverbs & Adjectives
    "not", "no", "yes", "up", "so", "out", "just", "now", "how", "then", "more",
    "also", "here", "well", "only", "very", "even", "back", "there", "down", "still",
    "good", "new", "first", "last", "long", "great", "little", "own", "other", "old",
    "right", "big", "high", "different", "small", "large", "next", "early", "young",
    "important", "few", "public", "bad", "same", "able", "nice", "fine", "cool",
    "sure", "true", "false", "fair", "real", "hard", "easy", "crazy", "fast", "slow",
    "best", "better", "worst", "damn", "fr", "literally", "totally", "definitely",
    "probably", "maybe", "obviously", "actually", "basically", "seriously", "honestly",
    "completely", "already", "soon", "late", "again", "together", "always", "never",
    "sometimes", "usually", "often", "ever", "anyway", "anyways", "awesome", "perfect",
    "done", "ready", "free", "busy", "simple", "clear", "clean", "huge", "tiny",
    "direct", "indirect", "general", "specific", "full", "empty", "previous",
    # Conversational & Slang
    "lol", "lmao", "rofl", "omg", "wtf", "plz", "pls", "thx", "thanks", "thank",
    "welcome", "sorry", "congrats", "bye", "hey", "hi", "hello", "yo", "sup",
    "yeah", "yep", "nope", "nah", "bro", "dude", "guy", "guys",
    # Common Tech, Work & Project Nouns
    "code", "project", "repo", "app", "web", "server", "backend", "frontend",
    "database", "api", "endpoint", "payload", "json", "xml", "curl", "header",
    "token", "branch", "pr", "bug", "issue", "error", "exception", "review",
    "system", "service", "client", "user", "data", "file", "link", "url", "page",
    "screen", "view", "call", "meet", "meeting", "interview", "placement", "company",
    "resume", "package", "round", "task", "job", "offer", "paper", "exam", "test",
    "question", "answers", "process", "time", "date", "status", "version", "result",
    "ticket", "flight", "carrier", "hotel", "travel", "fare", "price", "session",
    "id", "key", "common", "air", "inr", "ach", "cdta", "soap", "case", "class",
    "notes", "practice", "aim", "point", "matter", "sense", "stuff", "shit", "man",
    "chat", "msg", "message", "text", "mail", "email", "doc", "docs", "folder",
    "laptop", "pc", "phone", "wifi", "net", "tab", "browser", "team", "office",
    "career", "people", "thing", "things", "way", "day", "night", "week", "month",
    "year", "hour", "minute", "problem", "solution", "idea", "part", "place",
    "group", "world", "area", "side", "number",
    # Frameworks, languages, tools & infrastructure
    "angular", "react", "vue", "node", "next", "nuxt", "ts", "js", "python", "java",
    "cpp", "c", "rust", "golang", "go", "online", "offline", "site", "setup", "sync",
    "auth", "dev", "prod", "staging", "qa", "ui", "ux", "cli", "sdk", "lib",
    "framework", "stack", "git", "github", "gitlab", "docker", "k8s", "aws", "cloud",
    "linux", "mac", "windows", "feature", "release", "build", "pipeline", "sql",
    "mongo", "redis", "postgres", "rest", "graphql", "grpc", "html", "css", "yaml",
    "yml", "csv", "table", "chart", "graph", "cache", "cookie", "jwt", "oauth",
    "terminal", "powershell", "bash", "port", "ip", "dns", "domain", "ssl", "tls",
    "socket", "websocket", "ai", "ml", "nlp", "llm", "prompt", "bot", "model",
}

# Neutral / Universal tokens (programming syntax, entities, punctuation markers)
UNIVERSAL_TOKENS: Set[str] = {
    "url", "num", "email", "phone", "name", "date", "time", "deleted",
    "message", "omitted", "image", "audio", "video", "sticker", "contact",
    "vcard", "fn", "tel", "waid", "end", "begin", "version", "forwarded",
    "http", "https", "www", "com", "org", "net",
}

# Context-ambiguous homographs requiring adjacent-token smoothing
AMBIGUOUS_HOMOGRAPHS: Set[str] = {
    "he",   # Hindi copula ("wo bolra he") vs English pronoun ("he did it")
    "the",  # Hindi past copula ("log the") vs English article ("the code")
    "to",   # Hindi particle ("kal to chalenge") vs English prep ("go to")
    "me",   # Hindi locative ("room me") vs English pronoun ("tell me")
    "is",   # Hindi demonstrative ("is code me") vs English copula ("it is")
    "us",   # Hindi demonstrative ("us din") vs English pronoun ("with us")
    "in",   # Hindi demonstrative ("in logo") vs English prep ("in python")
    "so",   # Hindi verb ("jaake so") vs English adverb ("so cool")
    "no",   # Hindi number/particle vs English negation
}

RE_TOKENS = re.compile(r"[a-zA-Z\u0900-\u097F0-9_']+")


@dataclass
class TurnLanguageResult:
    """Language classification and code-switching metrics for a single turn."""

    total_tokens: int
    eng_tokens: int
    hin_tokens: int
    univ_tokens: int
    eng_ratio: float
    hin_ratio: float
    classification: str  # "pure_english", "pure_hindi", "code_switched", "neutral"
    switch_points: int
    switch_point_ratio: float
    token_tags: List[Tuple[str, str]]  # list of (token, tag)


@dataclass
class LanguageProfile:
    """Comprehensive language and code-switching profile for a corpus or archetype."""

    total_turns: int
    total_tokens: int

    # Overall Token Distribution
    token_distribution: Dict[str, float]

    # Turn-level Classification Breakdown
    turn_classification: Dict[str, float]

    # Code-Switching Dynamics
    switching_dynamics: Dict[str, Any]

    # Top Borrowed English Vocabulary in Hinglish
    borrowed_english_words: List[Dict[str, Any]]

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_turns": self.total_turns,
            "total_tokens": self.total_tokens,
            "token_distribution": self.token_distribution,
            "turn_classification": self.turn_classification,
            "switching_dynamics": self.switching_dynamics,
            "borrowed_english_words": self.borrowed_english_words,
            "metadata": self.metadata,
        }


class CodeSwitchingAnalyzer:
    """
    Analyzes code-switching dynamics, classifying token-level languages
    and quantifying turn-level language distributions and switch points.
    """

    def __init__(self):
        self.hindi_lexicon = set(HINDI_LEXICON)
        self.english_lexicon = set(ENGLISH_LEXICON)
        self.universal_tokens = set(UNIVERSAL_TOKENS)
        self.ambiguous_tokens = set(AMBIGUOUS_HOMOGRAPHS)

    def _classify_single_token(self, token: str) -> str:
        """Initial classification of a single lowercased token."""
        t = token.lower()

        # Check Universal / Numbers
        if t in self.universal_tokens or t.isdigit() or len(t) <= 1:
            return "UNIV"

        # Check Ambiguous
        if t in self.ambiguous_tokens:
            return "AMBIG"

        # Lexicon lookup
        in_hin = t in self.hindi_lexicon
        in_eng = t in self.english_lexicon

        if in_hin and not in_eng:
            return "HIN"
        if in_eng and not in_hin:
            return "ENG"

        # Morphological Heuristics for unlisted tokens
        # Check English inflections against English lexicon
        if t.endswith("s") and (t[:-1] in self.english_lexicon or (t.endswith("es") and t[:-2] in self.english_lexicon)):
            return "ENG"
        if t.endswith("ed") and (t[:-2] in self.english_lexicon or t[:-1] in self.english_lexicon):
            return "ENG"
        if t.endswith("ing") and (t[:-3] in self.english_lexicon or t[:-3] + "e" in self.english_lexicon):
            return "ENG"
        if t.endswith("ly") and (t[:-2] in self.english_lexicon or t[:-1] in self.english_lexicon):
            return "ENG"

        # Hindi verb / adjective inflections
        if t.endswith(("ra", "ri", "re", "rha", "rhi", "rhe", "ega", "egi", "enge", "unga", "ungi", "karo", "kare", "karte", "karta", "karti")):
            return "HIN"
        if t.startswith(("mat_", "kuch_", "bhai_")):
            return "HIN"

        # English suffixes
        if t.endswith(("ing", "tion", "sion", "ment", "able", "ible", "ize", "ized", "ity", "ous", "less", "ful", "ed", "ly")):
            return "ENG"

        # Default fallback based on character features
        return "ENG" if in_eng else "HIN"


    def classify_tokens(self, tokens: Sequence[str]) -> List[str]:
        """
        Classifies tokens with context-aware disambiguation for ambiguous homographs.
        """
        initial_tags = [self._classify_single_token(t) for t in tokens]
        resolved_tags = list(initial_tags)

        # Disambiguate homographs using context window of adjacent unambiguous tokens
        for i, tag in enumerate(initial_tags):
            if tag == "AMBIG":
                # Look at preceding and succeeding 2 tokens
                neighbors = []
                for offset in [-2, -1, 1, 2]:
                    idx = i + offset
                    if 0 <= idx < len(initial_tags) and initial_tags[idx] in {"ENG", "HIN"}:
                        neighbors.append(initial_tags[idx])

                if neighbors:
                    # Majority vote among neighbors
                    hin_count = sum(1 for n in neighbors if n == "HIN")
                    eng_count = sum(1 for n in neighbors if n == "ENG")
                    resolved_tags[i] = "HIN" if hin_count >= eng_count else "ENG"
                else:
                    # Default: in chat, words like 'he', 'to', 'me' are predominantly Hindi
                    resolved_tags[i] = "HIN"

        return resolved_tags

    def analyze_turn(self, text: str) -> TurnLanguageResult:
        """
        Analyzes a single conversational turn for language proportions and switch points.
        """
        raw_text = text.strip()
        tokens = [t.lower() for t in RE_TOKENS.findall(raw_text)]
        total_tokens = len(tokens)

        if total_tokens == 0:
            return TurnLanguageResult(
                total_tokens=0, eng_tokens=0, hin_tokens=0, univ_tokens=0,
                eng_ratio=0.0, hin_ratio=0.0, classification="neutral",
                switch_points=0, switch_point_ratio=0.0, token_tags=[],
            )

        tags = self.classify_tokens(tokens)
        token_tags = list(zip(tokens, tags))

        eng_count = sum(1 for tag in tags if tag == "ENG")
        hin_count = sum(1 for tag in tags if tag == "HIN")
        univ_count = sum(1 for tag in tags if tag == "UNIV")

        # Ratios among language-bearing content tokens
        content_tokens = eng_count + hin_count
        eng_ratio = round(eng_count / max(content_tokens, 1), 4)
        hin_ratio = round(hin_count / max(content_tokens, 1), 4)

        # Turn classification
        if content_tokens == 0:
            classification = "neutral"
        elif eng_count > 0 and hin_count == 0:
            classification = "pure_english"
        elif hin_count > 0 and eng_count == 0:
            classification = "pure_hindi"
        else:
            classification = "code_switched"

        # Calculate code-switch points (transitions between ENG and HIN)
        switch_points = 0
        last_lang = None
        for tag in tags:
            if tag in {"ENG", "HIN"}:
                if last_lang is not None and tag != last_lang:
                    switch_points += 1
                last_lang = tag

        switch_ratio = round(switch_points / max(content_tokens - 1, 1), 4)

        return TurnLanguageResult(
            total_tokens=total_tokens,
            eng_tokens=eng_count,
            hin_tokens=hin_count,
            univ_tokens=univ_count,
            eng_ratio=eng_ratio,
            hin_ratio=hin_ratio,
            classification=classification,
            switch_points=switch_points,
            switch_point_ratio=switch_ratio,
            token_tags=token_tags,
        )

    def fit(self, pairs: Sequence[Dict[str, Any]], name: str = "global") -> LanguageProfile:
        """
        Fits the CodeSwitchingAnalyzer on a sequence of turns to produce a complete LanguageProfile.
        """
        n_turns = len(pairs)
        if n_turns == 0:
            raise ValueError("Pairs sequence cannot be empty for language profiling.")

        total_tokens = 0
        total_eng = 0
        total_hin = 0
        total_univ = 0

        classification_counts: Counter[str] = Counter()
        switch_points_list: List[int] = []
        switched_turns_count = 0

        # Collect English words borrowed inside code-switched turns
        borrowed_english_counter: Counter[str] = Counter()

        for p in pairs:
            raw_text = p.get("target_text", "")
            turn_res = self.analyze_turn(raw_text)

            total_tokens += turn_res.total_tokens
            total_eng += turn_res.eng_tokens
            total_hin += turn_res.hin_tokens
            total_univ += turn_res.univ_tokens

            classification_counts[turn_res.classification] += 1
            switch_points_list.append(turn_res.switch_points)

            if turn_res.classification == "code_switched":
                switched_turns_count += 1
                # Collect the English words inserted into Hinglish
                for tok, tag in turn_res.token_tags:
                    if tag == "ENG" and len(tok) >= 3:
                        borrowed_english_counter[tok] += 1

        content_tokens = total_eng + total_hin

        token_dist = {
            "english_token_ratio": round(total_eng / max(content_tokens, 1), 4),
            "hindi_token_ratio": round(total_hin / max(content_tokens, 1), 4),
            "universal_token_ratio": round(total_univ / max(total_tokens, 1), 4),
            "total_english_tokens": total_eng,
            "total_hindi_tokens": total_hin,
            "total_universal_tokens": total_univ,
        }

        turn_dist = {
            "pure_english_ratio": round(classification_counts["pure_english"] / n_turns, 4),
            "pure_hindi_ratio": round(classification_counts["pure_hindi"] / n_turns, 4),
            "code_switched_ratio": round(classification_counts["code_switched"] / n_turns, 4),
            "neutral_ratio": round(classification_counts["neutral"] / n_turns, 4),
            "pure_english_turns": classification_counts["pure_english"],
            "pure_hindi_turns": classification_counts["pure_hindi"],
            "code_switched_turns": classification_counts["code_switched"],
            "neutral_turns": classification_counts["neutral"],
        }

        switch_dynamics = {
            "mean_switches_per_turn": round(float(np.mean(switch_points_list)), 2),
            "switched_turns_ratio": round(switched_turns_count / n_turns, 4),
            "max_switches_in_turn": int(np.max(switch_points_list)) if switch_points_list else 0,
        }

        borrowed_words = [
            {"word": w, "count": cnt} for w, cnt in borrowed_english_counter.most_common(20)
        ]

        return LanguageProfile(
            total_turns=n_turns,
            total_tokens=total_tokens,
            token_distribution=token_dist,
            turn_classification=turn_dist,
            switching_dynamics=switch_dynamics,
            borrowed_english_words=borrowed_words,
            metadata={"profile_name": name},
        )

    def analyze_by_style(
        self,
        pairs: Sequence[Dict[str, Any]],
        style_labels: np.ndarray,
        style_names: Optional[Dict[int, str]] = None,
    ) -> Dict[str, LanguageProfile]:
        """
        Computes language profiles broken down by communication archetype.
        """
        if len(pairs) != len(style_labels):
            raise ValueError("Lengths of pairs and style_labels must match.")

        unique_styles = sorted(np.unique(style_labels))
        profiles: Dict[str, LanguageProfile] = {}

        for s in unique_styles:
            s_mask = style_labels == s
            s_pairs = [pairs[i] for i in range(len(pairs)) if s_mask[i]]
            s_name = style_names.get(int(s), f"Style_{s:02d}") if style_names else f"Style_{s:02d}"
            profiles[s_name] = self.fit(s_pairs, name=s_name)

        return profiles
