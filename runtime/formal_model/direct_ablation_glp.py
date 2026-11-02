"""Matched ablations of the completed direct-emotion candidate. Not yet trained."""
import torch
from torch import nn
from dataclasses import replace
from formal_model.attention_graph import AttentionGraph
from formal_model.syntax_attention import SyntaxAttention
from formal_model.attention_candidate_glp import AttentionCandidateGLP

class DirectAblationGraph(AttentionGraph):
    arms=('none','direct','no_types','shuffle_types','mean_pool','no_role','no_time','no_emotion')
    ablation='direct'
    def pool(self,g,arm):
        mode={'no_types':'untyped_attention','shuffle_types':'shuffled_attention','mean_pool':'types_mean'}.get(self.ablation,'types_attention')
        return SyntaxAttention.pool(self,g,mode)
    def encode_words(self,g,arm='edges'):
        return self.pool(g,arm)[0]
    def graph_context(self,b,g,arm):
        x=self.utterance(b.utterances).tanh().float()
        if arm!='role_time':x=x+self.encode_words(g,self.syntax_mode)
        emo=self.graph_emotion(b.emotions.clamp_min(0))
        if self.ablation != 'no_emotion':x=x+emo
        nodes=[];edges=[([],[]) for _ in range(5)];positions=[];offset=0
        for i in range(len(x)):
            length=int(b.valid[i].sum());assert b.valid[i,:length].all() and not b.valid[i,length:].any()
            speakers=b.speakers[i,:length];unique=torch.unique(speakers,sorted=True)
            roles=self.role_kind((unique==b.target[i]).long())
            role_indices={int(s):offset+length+k for k,s in enumerate(unique)}
            nodes.extend([x[i,:length],roles,emo[i,:length]])
            positions.extend((i,j,offset+j) for j in range(length))
            for j in range(length):
                u=offset+j;r=role_indices[int(speakers[j])];e=offset+length+len(unique)+j
                for kind,a,z in ((0,u,r),(1,r,u)):
                    if self.ablation != 'no_role':
                        edges[kind][0].append(a);edges[kind][1].append(z)
                if j and self.ablation != 'no_time':edges[2][0].append(u-1);edges[2][1].append(u)
                if arm=='full':
                    edges[3][0].append(u);edges[3][1].append(e)
                    edges[4][0].append(e);edges[4][1].append(u)
            offset+=2*length+len(unique)
        h=torch.cat(nodes)
        for _ in range(2):
            message=self.graph_self(h)
            for layer,(source,dest) in zip(self.relations,edges):
                if not source:continue
                source=torch.tensor(source,device=h.device);dest=torch.tensor(dest,device=h.device)
                value=layer(h[source]).to(h.dtype);agg=torch.zeros_like(h);count=h.new_zeros(len(h))
                agg.index_add_(0,dest,value);count.index_add_(0,dest,h.new_ones(len(dest)))
                message=message+agg/count.clamp_min(1)[:,None]
            h=message.tanh()
        batch_i=torch.tensor([v[0] for v in positions],device=x.device)
        time_i=torch.tensor([v[1] for v in positions],device=x.device)
        node_i=torch.tensor([v[2] for v in positions],device=x.device)
        result=torch.zeros_like(x);result[batch_i,time_i]=h[node_i]
        return result

    def forward(self,b,prior,g,arm):
        assert arm in self.arms
        if arm=='none':return self.scaffold(b,prior,'coupled')
        self.ablation=arm
        context=self.graph_context(b,g,'direct_emotion')
        augmented=replace(b,utterances=b.utterances+self.output(context)*b.valid[:,:,None])
        return self.scaffold(augmented,prior,'coupled')

class DirectAblationGLP(AttentionCandidateGLP):
    arms=DirectAblationGraph.arms
    def __init__(self,backbone,dim=1024,classes=7,hidden=16):
        nn.Module.__init__(self)
        assert dim==1024 and classes==7
        self.backbone=backbone;self.classes=classes
        self.gold=nn.Embedding(classes,dim);nn.init.zeros_(self.gold.weight)
        self.candidate=DirectAblationGraph(hidden)
