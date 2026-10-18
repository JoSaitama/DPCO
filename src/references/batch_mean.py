"""Loss-reference gradients for the complete valid logical batch."""

from src.references.base import selected_mean


class BatchMeanReference:
    def selection_mask(self, valid_mask, feedback):
        return valid_mask

    def build(self, gradients, valid_mask, feedback, *, exclude_self=False):
        return selected_mean(gradients, valid_mask, exclude_self=exclude_self)
