import torch
import random
from tqdm import tqdm

from utils import AliasMultinomial

gpu_ids = []
if torch.cuda.is_available():
    gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
    device = torch.device(f'cuda:{gpu_ids[0]}')
    torch.cuda.set_device(device)
else:
    device = torch.device('cpu')

random.seed=3716

l = []

while sum(l) != 1:
    c = random.uniform(0,.12)
    if (sum(l) + c > 1):
        break
    l.append(c)

a = AliasMultinomial(probs=l, device=device)

for i in tqdm(range(10000000)):
    a.draw(15)
