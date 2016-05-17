import numpy as np
# start program from top level of project for import purposes
from src import net_initialize
from src.net_trainer import net_trainer

net = net_initialize.build()


trainer = net_trainer(net, learning_rate = 0.001)	

if net.pretrained:
	trainer.test(10)
else:
	trainer.train(2)
	net.save_network()
	print "~~ Networked saved as '%s'" % (net.net_name)
	print
	trainer.test(10)
