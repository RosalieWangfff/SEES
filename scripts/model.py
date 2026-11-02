"""Full SEES plus a relation-label permutation control with identical parameters.

Permutes existing edge labels within each utterance. Keeps actual dependency
connectivity, direction and label counts. Tests relation labels, NOT topology.
"""
import torch
from torch import nn
from recovery.model import ARMS as ORIGINAL_ARMS, PriorEvidenceModel as original

ARMS={**ORIGINAL_ARMS, 'G2__shuffle_rel':('gru','conditioned')}

def shuffled_relations(words, seed):
    rel=words['edge_relation']; result=rel.clone()
    child=words['edge_child']; utterance=words['word_to_utterance'][child]
    # Local indices make the control independent of minibatch packing/order.
    for u in torch.unique(utterance).tolist():
        ix=torch.where(utterance==u)[0]
        local=rel[ix].detach().cpu()
        generator=torch.Generator().manual_seed(seed+19100)
        perm=torch.randperm(len(local),generator=generator)
        result[ix]=local[perm].to(rel.device)
    return {**words,'edge_relation':result}

class ShuffledRelations(nn.Module):
    def __init__(self, structure, seed):
        super().__init__()
        # Keep identical state_dict names and parameter budget.
        self.message=structure.message; self.relation=structure.relation
        self.output=structure.output; self.seed=seed
    def forward(self,h,words):
        from formal_model.english_structure_matrix.structure import LocalStructure
        words=shuffled_relations(words,self.seed)
        self.mode='dep_typed'
        return LocalStructure.forward(self,h,words)

def PriorEvidenceModel(text,dim,classes,arm,scaffold,seed):
    model=original(text,dim,classes,'G2__dep_typed' if arm=='G2__shuffle_rel' else arm,scaffold,seed)
    if arm=='G2__shuffle_rel':
        model.evidence.structure=ShuffledRelations(model.evidence.structure,seed)
    return model
