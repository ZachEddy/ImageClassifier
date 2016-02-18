import numpy as np
# start program from top level of project for import purposes
from src import net_initialize
net_initialize

# I need to these comments... but I don't want to get rid of them. Just in case. I guess that makes me a hoarder?
# 
# 
# IF YOU HAVE 140 LINES OF COMMENTED CODE... 
# ... YOU MIGHT BE A TERRIBLE PROGRAMMER

# a = np.array([[[1,2,3],[4,6,7],[4,1,6]], [[3,5,3],[5,7,2],[6,1,5]], [[5,6,0],[9,2,4],[2,5,5]]])
# # print a[0][1]
# # for i in range(len(a)):
# # 	print "he" #np.multiply(a[i,::], [[1,2]])


# # print a 



# stride = 2


# input_volume = np.array([[0,0,0,0,0,0,0,0,2,0,2,1,1,0,0,2,1,0,0,0,0,0,0,2,2,2,1,0,0,0,2,1,1,0,0,0,1,2,2,1,1,0,0,0,0,0,0,0,0],
# 	[0,0,0,0,0,0,0,0,1,1,0,2,0,0,0,1,0,2,0,1,0,0,2,0,1,0,0,0,0,2,0,2,2,1,0,0,1,2,2,0,0,0,0,0,0,0,0,0,0],
# 	[0,0,0,0,0,0,0,0,2,0,2,2,1,0,0,0,2,2,2,2,0,0,2,2,2,1,0,0,0,2,1,2,2,0,0,0,2,0,1,1,1,0,0,0,0,0,0,0,0]]
# 	)

# input_volume = np.reshape(input_volume,(3,7,7))

# for i in input_volume:
# 	i = np.transpose(i)


# filters = np.array([1,0,-1,-1,0,1,1,1,0,1,0,0,-1,0,-1,1,0,0,0,0,-1,-1,1,0,1,1,1])
# filters = np.reshape(filters,(3,3,3))

# for i in range(len(filters)):
# 	filters[i] = np.transpose(filters[i])


# for i in range(len(input_volume)):
# 	input_volume[i] = input_volume[i].T




# # ty = range(10)

# # for i in ty:
# # 	i = 2 * i

# # print ty

# # print input_volume[:,1:3,0:2]

# output = np.array([])
# for i in range(3):
# 	row = i * stride
				
# 	# for all the rows, go through all the columns
# 	for j in range(3):
# 		col = j * stride
# 		inputs = input_volume[:, row:row+3, col:col+3]
# 		conv_calc = np.sum(inputs * filters) + 1
# 		output = np.append(output,conv_calc)

# output = output.T
# output = np.reshape(output, (3,3))
# print output



# 		# depth_column = volume[:,row + stride, col + stride]
# 		# conv_calculation = np.sum(depth_column * filter_volume)

# 		# output_volume = np.append(output_volume, conv_calculation)





# # print filters







# # input_vol = [0,0,0,0,0,0,0,0,0,1,0,2,1,0,0,2,2,1,1,0,0,0,2,2,0,1,0,0,0,0,0,0,1,0,0,0,1,1,2,1,0,0,0,0,0,0,0,0,0,
# # 			0,0,0,0,0,0,0,0,2,2,1,1,1,0,0,2,0,0,2,1,0,0,1,2,1,2,0,0,0,1,1,0,0,0,0,0,1,0,0,2,0,0,0,0,0,0,0,0,0,
# # 			0,0,0,0,0,0,0,0,2,0,2,0,2,0,0,0,1,1,0,0,0,0,2,1,0,0,2,0,0,1,0,2,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]

# # 		input_vol = np.reshape(input_vol, (3,7,7))




# # a = np.random.random((4,3,3))
# # b = np.random.random((3, 3))

# # print a
# # print ""
# # print ""
# # print ""
# # print ""
# # print ""
# # a = np.append(a,b)
# # # a = np.reshape(a, (,3,3))
# # print a

# # print a.shape




# # a = np.append(a,b,2)

# # print a

# # c = np.random.random((4, 4))
# # d = np.dstack([a,b,c])

# # print d



# # # print np.sum(a * a)


# # # print type(a)

# # # toAdd = [[1,2,8],[1,5,8],[1,5,7]]


# # # # a = np.stack((a,toAdd))

# # # # a = np.append(a,toAdd,2)

# # # print a[0].ndim
# # # # print len(a)
# # # print a
# # # # print "..."
# # # # print a[:,0:1,0:2]