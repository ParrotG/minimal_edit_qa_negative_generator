from .base import CandidateFilter, FilterDecision
from .basic import AnswerTypeFilter, EditDistanceFilter, LengthRatioFilter
from .nli_flip import NLIFlipFilter

DEFAULT_FILTERS = [
    LengthRatioFilter(),
    EditDistanceFilter(),
    AnswerTypeFilter(),
]
