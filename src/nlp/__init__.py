"""
NLP package for persona analysis and feature extraction.
"""

from src.nlp.behavior import BehaviorDimension, BehavioralExtractor, BehavioralProfile
from src.nlp.clustering import ClusterConfig, ClusteringEngine, ClusteringResult
from src.nlp.code_switching import CodeSwitchingAnalyzer, LanguageProfile
from src.nlp.discourse import DiscourseAct, DiscourseAnalyzer, DiscourseProfile, TurnDiscourseResult
from src.nlp.embeddings import EmbeddingConfig, EmbeddingEngine
from src.nlp.linguistics import LinguisticProfile, LinguisticProfiler
from src.nlp.similarity import SimilarityEngine, SimilarityMatch
from src.nlp.situations import SituationalCategory, SituationalClassifier, SituationalProfile, SituationResult
from src.nlp.syntax import SyntacticProfile, SyntacticProfiler
from src.nlp.topics import TopicConfig, TopicDiscoveryEngine, TopicModelResult

__all__ = [
    "BehaviorDimension",
    "BehavioralExtractor",
    "BehavioralProfile",
    "ClusterConfig",
    "ClusteringEngine",
    "ClusteringResult",
    "CodeSwitchingAnalyzer",
    "DiscourseAct",
    "DiscourseAnalyzer",
    "DiscourseProfile",
    "EmbeddingConfig",
    "EmbeddingEngine",
    "LanguageProfile",
    "LinguisticProfile",
    "LinguisticProfiler",
    "SimilarityEngine",
    "SimilarityMatch",
    "SituationalCategory",
    "SituationalClassifier",
    "SituationalProfile",
    "SituationResult",
    "SyntacticProfile",
    "SyntacticProfiler",
    "TopicConfig",
    "TopicDiscoveryEngine",
    "TopicModelResult",
    "TurnDiscourseResult",
]





