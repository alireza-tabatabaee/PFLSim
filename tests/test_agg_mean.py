# tests/test_agg_mean.py
import torch
import torch.nn as nn
import pflsim
from pflsim.strategy import FedAvg

class W(nn.Module):
    def __init__(self): 
        super().__init__(); 
        self.p = nn.Parameter(torch.zeros(2,3))
    def forward(self, x):
        return x @ self.p.t()

def setp(m, val): 
    with torch.no_grad(): 
        m.p.copy_(torch.tensor(val, dtype=torch.float32))

pfl = pflsim.PFLSim(3)
pfl._STRATEGY = FedAvg()
def test_fedavg_exact_mean():
    m1, m2, m3 = W(), W(), W()
    setp(m1, [[1,2,3],[4,5,6]])
    setp(m2, [[7,8,9],[10,11,12]])
    setp(m3, [[-1,0,1],[0,0,0]])

    pfl.begin_round()

    for m in (m1,m2,m3): 
        pfl.send(m)
    pfl.aggregate()

    mg = W(); pfl.load_global(mg)
    expected = (m1.state_dict()["p"] + m2.state_dict()["p"] + m3.state_dict()["p"]) / 3
    assert torch.allclose(mg.state_dict()["p"], expected, atol=0, rtol=0)
