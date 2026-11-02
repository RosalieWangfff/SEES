"""Durable progress checkpoints and cooperative interruption for long runs."""
import os
import signal
import time

import numpy as np
import torch

from .common import ROOT


STOP_FILE = ROOT / 'STOP_REQUESTED'


class StopController:
    def __init__(self, minutes=30):
        if minutes <= 0:
            raise ValueError('checkpoint minutes must be positive')
        self.interval = minutes * 60
        self.last_save = time.monotonic()
        self.requested = False
        self.previous = {}
        for name in ('SIGINT', 'SIGTERM', 'SIGBREAK'):
            signum = getattr(signal, name, None)
            if signum is not None:
                self.previous[signum] = signal.getsignal(signum)
                signal.signal(signum, self._request)

    def _request(self, signum, frame):
        self.requested = True
        print('Stop requested; finishing the current optimizer step, then saving.', flush=True)

    def stop_requested(self):
        return self.requested or STOP_FILE.exists()

    def save_due(self):
        return time.monotonic() - self.last_save >= self.interval

    def saved(self):
        self.last_save = time.monotonic()

    def acknowledge_stop(self):
        STOP_FILE.unlink(missing_ok=True)
        print('Progress saved. Re-run the same command to resume.', flush=True)

    def close(self):
        for signum, previous in self.previous.items():
            signal.signal(signum, previous)


def rng_state():
    return dict(python=__import__('random').getstate(), numpy=np.random.get_state(),
                cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    __import__('random').setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    torch.cuda.set_rng_state_all(state['cuda'])


def sync_file(path):
    """Flush a feature cache before publishing the cursor that depends on it."""
    with open(path, 'r+b') as f:
        os.fsync(f.fileno())
