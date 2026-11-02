"""Paired SEES controls. Retain prior-only slice of evidence output projection."""
import types
import torch
from nc_revision.model import ARMS as ORIGINAL_ARMS, PriorEvidenceModel as original

ARMS = {'G2__dep_typed': ORIGINAL_ARMS['G2__dep_typed'],
        'G2__no_evidence': ORIGINAL_ARMS['G2__dep_typed']}


def no_word_evidence(self, words, prior_state, own_role, conditioned):
    # output([evidence, prior_state]) retains the prior half, bias and dimensions.
    # Token features, structure, and both attention levels do not affect output.
    zero = torch.zeros_like(prior_state)
    correction = self.output(torch.cat((zero, prior_state), dim=-1))
    alpha = prior_state.new_zeros((len(prior_state), words['lookup'].shape[1]))
    return correction, alpha


def PriorEvidenceModel(text, dim, classes, arm, scaffold, seed):
    if arm not in ARMS:
        raise ValueError(arm)
    model = original(text, dim, classes, 'G2__dep_typed', scaffold, seed)
    model._remove_word_evidence = (arm == 'G2__no_evidence')
    return model


def install_control(model):
    from formal_model.english_predicted.mixed_runtime import install_model
    model = install_model(model)
    if model._remove_word_evidence:
        model.evidence.forward = types.MethodType(no_word_evidence, model.evidence)
    return model
