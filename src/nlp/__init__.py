"""
NLP package for persona analysis and feature extraction.
"""

from src.nlp.clustering import ClusterConfig, ClusteringEngine, ClusteringResult
from src.nlp.code_switching import CodeSwitchingAnalyzer, LanguageProfile
from src.nlp.embeddings import EmbeddingConfig, EmbeddingEngine
from src.nlp.linguistics import LinguisticProfile, LinguisticProfiler
from src.nlp.similarity import SimilarityEngine, SimilarityMatch
from src.nlp.syntax import SyntacticProfile, SyntacticProfiler
from src.nlp.topics import TopicConfig, TopicDiscoveryEngine, TopicModelResult

__all__ = [
    "ClusterConfig",
    "ClusteringEngine",
    "ClusteringResult",
    "CodeSwitchingAnalyzer",
    "EmbeddingConfig",
    "EmbeddingEngine",
    "LanguageProfile",
    "LinguisticProfile",
    "LinguisticProfiler",
    "SimilarityEngine",
    "SimilarityMatch",
    "SyntacticProfile",
    "SyntacticProfiler",
    "TopicConfig",
    "TopicDiscoveryEngine",
    "TopicModelResult",
]





