# PFLSim

A simple torch-based library to evaluate various regular federated learning and personalized federated learning schemes. 

In particular it's written in a way to allow the user to use the functions in a "drop-in" manner with regular non-federated code to 1) convert regular pytorch training code to federated simulation with minimal hassle, and 2) Allow the user to switch between regular federated schemes with just changing one line of code.

## Implemented schemes
✅ done, ❌ not done, 🛠️ WIP

**Regular federated learning**
| # | Algorithm                               | State | Reference | 
|---|------------------------------------|:------:|:------:|
| 1 | FedAvg                  |   ✅     |        | 
| 2 | FedProx                 |   ✅     |        |
| 3 | FedNova                 |   🛠️     |        |

**Personalized schemes**
| # | Algorithm                               | State | Reference | 
|---|------------------------------------|:------:|:------:|
| 1 | PerFedAvg               |   🛠️     |        | 
| 2 | pFedMe                  |   🛠     |        |
| 3 | DITTO                   |   ❌     |        |

