"""SeqLoRA, the sequential-finetuning (SFT) baseline: a single LoRA adapter
trained across every task in turn, with no regularizer, replay, or per-task
routing, used for every class at inference."""

from models.lora import Learner as LoRALearner


class Learner(LoRALearner):
    def __init__(self, args):
        super().__init__(args)
        # single adapter -> nothing to merge/accumulate
        self.train_merge = False
        self._network.merge = False

    def _train_adapter(self):
        return 0
