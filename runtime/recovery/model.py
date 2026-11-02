from formal_model.core_prior_ablation.model import PriorEvidenceModel as Base, ARMS as BASE_ARMS
from formal_model.english_structure_matrix.model import PriorEvidenceModel as Structured, ARMS as STRUCT_ARMS
ARMS={**{k:v for k,v in BASE_ARMS.items() if k in ('G1','G_word','G2')},**STRUCT_ARMS}
def PriorEvidenceModel(text,dim,classes,arm,scaffold,seed):
    return (Structured if '__' in arm else Base)(text,dim,classes,arm,scaffold,seed)
