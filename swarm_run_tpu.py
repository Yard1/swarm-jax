import functools
import multiprocessing

import optax
import ray
import traceback

from loader import TextLoader
from ray_tpu import start_ray, get_connection, create_tpu, wait_til, delete_tpu
from swarm_jax.model import SwarmCharTransformerBig
from swarm_jax.swarm import Swarm
from swarm_jax.swarm_layer import NetworkPrecision

if __name__ == '__main__':
    tpus = 2
    zone = "europe-west4-a"

    # for i in range(tpus):
    #     delete_tpu(f"swarm-jax-test-{i}", zone)

    # exit()

    head_info = ray.init(dashboard_host="0.0.0.0")
    address = head_info['redis_address']

    conns = []
    for i in range(tpus):
        create_tpu(f"swarm-jax-test-{i}", zone, "v3-8", False)

    try:
        for i in range(tpus):
            assert wait_til(f"swarm-jax-test-{i}", zone, {'state': 'READY', 'health': 'HEALTHY'})

        for i in range(tpus):
            conns += get_connection(f"swarm-jax-test-{i}", zone)

        with multiprocessing.Pool(processes=tpus) as p:
            p.map(functools.partial(start_ray, address=address), conns)

        train_dataset = TextLoader("data/enwik8", batchsize=(1, 16), sample_size=128, length=90000000)

        optimizer = optax.chain(
            optax.clip_by_global_norm(0.25),
            optax.adam(2e-4, b1=0.9, b2=0.99, eps=1e-5))

        prec = NetworkPrecision(fwd_act="float32", rev_act="float32", grad="float32")

        model = SwarmCharTransformerBig
        swarm = Swarm(model, optimizer, 2 ** 16, train_dataset.get_samples, prec)
        swarm.run(100000, "runs/512_30L", "ckpt/512_30L")
    except Exception:
        traceback.print_exc()
        for i in range(tpus):
            delete_tpu(f"swarm-jax-test-{i}", zone)
    ray.shutdown()
