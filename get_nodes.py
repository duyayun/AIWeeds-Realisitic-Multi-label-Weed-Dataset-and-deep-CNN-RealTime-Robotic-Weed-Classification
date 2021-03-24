import tensorflow as tf
with tf.Session() as sess:
    with open('./trt_resnet_sigmoid_16epoch.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read()) 
        print (graph_def)
