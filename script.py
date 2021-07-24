from collections import Counter
import socket
import time

import ray

ray.init(address='auto')


print('''This cluster consists of
    {} nodes in total
    {} CPU resources in total
'''.format(len(ray.nodes()), ray.cluster_resources()))

@ray.remote(resources={"TPU": 1})
def f():
    time.sleep(0.1)
    # Return IP address.
    print("I am running on a TPU!")
    return socket.gethostbyname(socket.gethostname())

object_ids = [f.remote() for _ in range(100)]
ip_addresses = ray.get(object_ids)

print('Tasks executed')
for ip_address, num_tasks in Counter(ip_addresses).items():
    print('    {} tasks on {}'.format(num_tasks, ip_address))