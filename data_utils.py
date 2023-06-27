import os
from PIL import Image
import numpy as np

def load_data(ids, im_shape, im_shape_final, train_path):
    X = []
    Y = []
    
    for i in ids:
        y = Image.open(os.path.join(train_path, i))
        x = np.array(y.resize(im_shape).resize(im_shape_final))/255.0
        y = np.array(y.resize(im_shape_final))/255.0
        if len(y.shape)<3:
            y = np.expand_dims(y,2)
            y = np.concatenate([y,y,y],axis=2)
            
            x = np.expand_dims(x,2)
            x = np.concatenate([x,x,x],axis=2)
            
        X.append(x)
        Y.append(y)
        
    return np.array(X), np.array(Y)

def batch_generator(ids, im_shape, im_shape_final, train_path, batch_size = 8):
    batch = []
    while True:
        np.random.shuffle(ids)
        for i in ids:
            batch.append(i)
            if len(batch)==batch_size:
                yield load_data(batch, im_shape, im_shape_final, train_path)
                batch = []