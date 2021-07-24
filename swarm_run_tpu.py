import optax
import ray
import traceback

from swarm_jax.model import SwarmCharTransformerBig
from swarm_jax.swarm import Swarm
from swarm_jax.swarm_layer import NetworkPrecision

import os
import time

import mmap
import numpy as np


class TextLoader():
    def __init__(self, fname, batchsize, sample_size, offset=0, length=0):
        self.f = open(fname, "r+b")
        self.mm = mmap.mmap(self.f.fileno(), length=length, offset=offset)
        self.file_size = os.stat(fname).st_size
        self.bs = np.product(batchsize)

        if isinstance(batchsize, tuple):
            self.batch_shape = batchsize
        else:
            self.batch_shape = (batchsize,)
        self.ss = sample_size

        self.np_mm = np.memmap(fname, dtype='uint8', mode='r', shape=(self.file_size,))

    def get_samples(self):
        sample = np.random.randint(0, self.file_size - 2 - self.ss, self.bs)
        batch = np.zeros((self.bs, self.ss + 1))

        for i in range(self.ss + 1):
            batch[:, i] = self.np_mm[sample + i]

        target = batch[:, 1:].astype(np.uint32)
        target = target.reshape(self.batch_shape + (self.ss,))

        obs = batch[:, :-1].astype(np.uint32)
        obs = obs.reshape(self.batch_shape + (self.ss,))

        return {"target": target, "obs": obs}

if __name__ == '__main__':
    ray.init(address='auto')

    train_dataset = TextLoader("swarm-jax/data/enwik8", batchsize=(8, 8), sample_size=1024, length=90000000)

    optimizer = optax.chain(
        optax.clip_by_global_norm(0.25),
        optax.adam(2e-4, b1=0.9, b2=0.99, eps=1e-5))

    prec = NetworkPrecision(fwd_act="float32", rev_act="float32", grad="float32")

    model = SwarmCharTransformerBig
    swarm = Swarm(model, optimizer, 2 ** 16, train_dataset.get_samples, prec)
    swarm.run(100000, "runs/512_30L", "ckpt/512_30L")
