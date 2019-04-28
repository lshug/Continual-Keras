#foreach output,label pair, get gradient list and 
        #elems=(wrapped_model.output,label)
        #gradient_map = K.map_fn(lambda x: K.gradients(categorical_nll(x[1],x[0]),wrapped_model.trainable_weights), elems, dtype=[tf.float32 for x in range(len_weights)]) 
        #gradients = K.get_session().run(gradient_map, feed_dict={wrapped_model.input:X})