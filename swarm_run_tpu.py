import optax
import ray
import traceback

from loader import TextLoader
from swarm_jax.model import SwarmCharTransformerBig
from swarm_jax.swarm import Swarm
from swarm_jax.swarm_layer import NetworkPrecision

if __name__ == '__main__':
    ray.init(address='auto')

    try:
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
    ray.shutdown()
