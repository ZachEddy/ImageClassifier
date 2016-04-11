import numpy as np
# start program from top level of project for import purposes
from src import net_initialize
from src.net_trainer import net_trainer

net = net_initialize.build()

net_trainer(net, learning_rate = 0.001)