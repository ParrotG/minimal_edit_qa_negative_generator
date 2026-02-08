from .base import CandidateFilter, FilterDecision
from .basic import AnswerTypeFilter, EditDistanceFilter, LengthRatioFilter
from .grammar import GrammarFilter
from .qa_consistency import QAConsistencyFilter
from .nli_flip import NLIFlipFilter

DEFAULT_FILTERS = [
    LengthRatioFilter(),
    EditDistanceFilter(),
    AnswerTypeFilter(),
]
