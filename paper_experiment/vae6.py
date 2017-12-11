def mnist_vae(data,gene_size,feed_dict):
    import numpy as np
    from vae4 import mnist_vae
    seper = []
    for i in range(data.shape[1]):
    	seper.append(len(set(data[:,i])))
    mask = np.array(seper)>2
    #    print(seper)
    #    mask.transpose()
    continuous = data[:,mask]
    x_hat_1 = mnist_vae(continuous,gene_size,feed_dict)
    from sklearn.neighbors import KNeighborsClassifier 
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(continuous,range(len(continuous)))
    com = neigh.predict(x_hat_1)
    pre = []
    i = 0
    for index,value in enumerate(mask):
    	if value == True:
    		pre.append(x_hat_1[:,i])
    		i = i+1
    	else:
    		tmp = data[com,index]
    		pre.append(tmp)
    pre = np.array(pre)
    return pre.transpose()