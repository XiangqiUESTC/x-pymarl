from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .decompose_learner import DecomposeLearner

REGISTRY = {
    "q_learner": QLearner,
    "coma_learner": COMALearner,
    "qtran_learner": QTranLearner,
    "decompose_learner": DecomposeLearner,
}

