"""One CPU check for changed model/control and generated execution paths."""
import argparse,ast,importlib.util,sys
from pathlib import Path
import torch
from prepare import generate

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();sys.path.insert(0,str(root))
    files=generate(root)
    for n,s in files.items():ast.parse(s,filename=n)
    # Inspect CLI generation without importing/initializing GPU trainers.
    assert 'seed(1008)' not in files['glp_train.py']
    assert '1008*100+epoch' not in files['glp_train.py']
    assert 'seed=a.seed' in files['glp_train.py']
    assert "ROOT/'test_results_nc_revision'" in files['evaluate.py']
    spec=importlib.util.spec_from_file_location('revision_model',Path(__file__).with_name('model.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    from formal_model.english_structure_matrix.structure import LocalStructure
    torch.manual_seed(42)
    base=LocalStructure('dep_typed');control=m.ShuffledRelations(base,42)
    assert set(base.state_dict())==set(control.state_dict())
    assert sum(p.numel() for p in base.parameters())==sum(p.numel() for p in control.parameters())
    # Synthetic tensors are only a software check, not scientific evidence.
    words={'edge_child':torch.tensor([0,1,2,4,5,6]),'edge_parent':torch.tensor([1,2,3,5,6,7]),
           'edge_relation':torch.tensor([1,2,3,4,5,6]),'edge_negation':torch.zeros(6,dtype=torch.bool),
           'word_to_utterance':torch.tensor([0]*4+[1]*4),'valid':torch.ones(8,dtype=torch.bool)}
    changed=m.shuffled_relations(words,42)
    assert torch.equal(words['edge_child'],changed['edge_child'])
    assert torch.equal(words['edge_parent'],changed['edge_parent'])
    for sl in (slice(0,3),slice(3,6)):
        assert torch.equal(words['edge_relation'][sl].sort().values,changed['edge_relation'][sl].sort().values)
    h=torch.randn(8,32,requires_grad=True)
    assert torch.equal(control(h,words),h) # Preserve zero initialization.
    torch.nn.init.normal_(control.output.weight,std=.05)
    result=control(h,words);result.square().sum().backward()
    assert torch.isfinite(result).all() and torch.isfinite(h.grad).all()
    assert control.relation.weight.grad.abs().sum()>0
    # Same utterance must receive same perturbation when packed alone.
    single={k:v.clone() for k,v in words.items()}
    for k in ('edge_child','edge_parent','edge_relation','edge_negation'):single[k]=single[k][3:]
    single['edge_child']-=4;single['edge_parent']-=4
    single['word_to_utterance']=torch.zeros(4,dtype=torch.long);single['valid']=torch.ones(4,dtype=torch.bool)
    assert torch.equal(m.shuffled_relations(single,42)['edge_relation'],changed['edge_relation'][3:])
    assert torch.allclose(control(h[4:],single),result[4:],atol=1e-6)
    print('PASS: generated source, configurable seeds, output isolation, control parameter parity, gradients and batch invariance. No training or efficacy result.')

if __name__=='__main__':main()
