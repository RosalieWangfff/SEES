"""Occurrence-aligned predicted labels; target labels and original data unchanged."""
import json
from pathlib import Path
import numpy as np
from formal_model.core_prior.data import CoreData as GoldData, ROOT
from formal_model.train_extension_extractor import digest


class CoreData(GoldData):
    def __init__(self, dataset, need_text):
        assert dataset in ('meld','dailydialog','emorynlp')
        super().__init__(dataset, need_text)
        folder = ROOT/'results_predicted_english'/dataset/'cache'
        done = json.loads((folder/'complete.json').read_text(encoding='utf-8'))
        assert done['complete'] and done['full_model_oof'] and not done['diagnostic_only']
        assert done['labels'] == self.labels and not done['test_used']
        assert digest(folder/'history_predictions.npz') == done['cache_sha']
        assert digest(folder/'erc_protocol.json') == done['protocol_sha']
        self.hashes['predicted_history_cache'] = done['cache_sha']
        self.hashes['erc_protocol'] = done['protocol_sha']
        from formal_model.english_predicted.english_data import aligned
        _, _, aliases = aligned(ROOT,dataset)
        with np.load(folder/'history_predictions.npz', allow_pickle=False) as z:
            for split in ('train', 'val'):
                assert self.hashes[split] == done['prepared_sources'][split]
                keys, pred = z[split+'_keys'], z[split+'_pred']
                assert len(set(keys.tolist())) == len(keys) == len(pred)
                assert ((pred >= 0) & (pred < self.classes)).all()
                lookup = dict(zip(keys.tolist(), pred.tolist()))
                for row in self.rows[split]:
                    row['emotions'] = [lookup[key] for key in aliases[split][str(row['id'])]]
