#!/disk/scratch/mlp/miniconda2/bin/python
import json, os, sys, time
import numpy as np
import tensorflow as tf
#from sklearn.metrics import confusion_matrix
from data_providers import MSDGenreDataProvider

MODELDIR='%s/'
LOGFILE='%s/log.json'
PARAMFILE='%s/params.json'


def create_kaggle_submission_file(pred_classes, output_file, overwrite=False):
    if os.path.exists(output_file) and not overwrite:
        raise ValueError('File already exists at {0}'.format(output_file))
    #pred_classes = predictions.argmax(-1)
    ids = np.arange(pred_classes.shape[0])
    np.savetxt(output_file, np.column_stack([ids, pred_classes]), fmt='%d',
               delimiter=',', header='Id,Class', comments='')

def normpdf(x, mu, sigma):
    u = (x-mu)/np.abs(sigma)
    y = np.exp(-u*u*0.5)/(np.sqrt(2.*np.pi)*abs(sigma))
    return y

def get_activation(params):
    return (
        tf.nn.tanh,
        tf.nn.sigmoid,
        tf.nn.relu,
        tf.nn.relu6,
        tf.nn.elu)[
        ('tanh',
        'sigmoid',
        'relu',
        'relu6',
        'elu').index(params['activation'])
    ]

def get_optimizer(params):
    if params['optimizer']=='GradientDescentOptimizer':
        return tf.train.GradientDescentOptimizer(params['learning_rate'])
    return (
        tf.train.GradientDescentOptimizer,
        tf.train.RMSPropOptimizer,
        tf.train.AdamOptimizer,
        tf.train.AdagradOptimizer,
        tf.train.AdadeltaOptimizer)[
        ('GradientDescentOptimizer',
        'RMSPropOptimizer',
        'AdamOptimizer',
        'AdagradOptimizer',
        'AdadeltaOptimizer').index(params['optimizer'])
    ](params['learning_rate'],beta1=params['beta1'],beta2=params['beta2'])

def get_rnn_cell(params,layer):
    if params['rnn_cell'] == 'basic_rnn':
        return tf.contrib.rnn.BasicRNNCell(params['rnn_dims'][layer],activation=get_activation(params))
    elif params['rnn_cell'] == 'basic_lstm':
        return tf.contrib.rnn.BasicLSTMCell(params['rnn_dims'][layer],activation=get_activation(params),forget_bias=params['forget_bias'],state_is_tuple=True)
    elif params['rnn_cell'] == 'lstm':
        return tf.contrib.rnn.LSTMCell(params['rnn_dims'][layer],activation=get_activation(params),forget_bias=params['forget_bias'],use_peepholes=params['peepholes'],state_is_tuple=True)
    elif params['rnn_cell'] == 'gru':
        return tf.contrib.rnn.GRUCell(params['rnn_dims'][layer],activation=get_activation(params))

def get_weight_init(params):
    if params['weight_init'] == 'glorot':
        return lambda shape: tf.truncated_normal(shape,stddev=params['weight_glorot_gain']*np.sqrt(2./sum(shape)))
    elif params['weight_init'] == 'uniform':
        return lambda shape: tf.random_uniform(shape,-params['weight_uniform_gain'],params['weight_uniform_gain'],dtype=tf.float32)
    elif params['weight_init'] == 'normal':
        return lambda shape: tf.truncated_normal(shape, stddev=params['weight_normal_gain'])
def get_conv_weight_init(params):
    return lambda shape: tf.truncated_normal(shape, stddev=params['weight_normal_gain'])

def weight_variable(shape,weight_init,name='W'):
    return tf.Variable(weight_init(shape),name=name)

def bias_variable(shape,name='b'):
    return tf.Variable(tf.constant(0.0, shape=shape),name=name)

def fc_layer(x,size,weight_init,activation=tf.nn.relu,names=['W','b','a']):
    W = weight_variable([int(x.get_shape()[1]),size],weight_init,name=names[0])
    b = bias_variable([size],name=names[1])
    a = activation(tf.nn.xw_plus_b(x, W, b,name='z'),name=names[2])
    return W, b, a

def conv(x, W, padding):
    return tf.nn.conv1d(x, W, stride=1, padding=padding)

def maxpool(x, size):
    return tf.nn.pool(x, [size], 'MAX', 'VALID',strides=[size])

def conv_layer(x,shape,weight_init,padding='SAME',activation=tf.nn.relu,names=['W','b','a']):
    W = weight_variable(shape,weight_init,name=names[0])
    b = bias_variable([shape[-1]],name=names[1])
    a = activation(conv(x, W, padding) + b, name=names[2])
    return W, b, a

def create_input_layers(params,layers,inputs_shape):
    with tf.name_scope('inputs'):
        inputs = tf.placeholder(tf.float32,shape=(None,)+inputs_shape[1:],name='inputs')
        layers.append(tf.reshape(inputs,[-1,inputs_shape[-1]//25,25]))

        if 'deltas' in params['features']:
            inputs_deltas = (layers[-1][:,2:,:]-layers[-1][:,:-2,:])*0.5
            layers.append(tf.concat([layers[-1][:,1:-1,:],inputs_deltas],axis=2, name='deltas'))

        if params['gaussian_noise']>0.0:
            layers.append(layers[-1] + tf.random_normal(shape=tf.shape(layers[-1]), mean=0.0, stddev=params['gaussian_noise'], dtype=tf.float32,name='gaussian_noise'))

    return inputs

def create_fc_layers(params,layers):
    fc_weights = []
    for i in range(len(params['fc_dims'])):
        with tf.name_scope('fc-%d'%(i+1)):
            W, b, a = fc_layer(layers[-1], params['fc_dims'][i], get_weight_init(params), get_activation(params))
            layers.append(a)
            fc_weights.append(W)
    return fc_weights

def create_output_layer(params,layers,num_classes):
    with tf.name_scope('output'):
        W, b, a = fc_layer(layers[-1], num_classes, get_weight_init(params), tf.identity)
        layers.append(a)

def create_eval_and_train(params,layers,inputs_shape,num_classes,fc_weights):
    outputs = layers[-1]

    with tf.name_scope('eval'):
        targets = tf.placeholder(tf.float32,shape=(None,num_classes),name='targets')
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(targets, 1), tf.argmax(outputs, 1)), tf.float32),name='accuracy')

        ll2 = tf.nn.log_softmax(outputs)
        if params['gaussian_filter_std']:
            gf = normpdf(np.array([-2, -1, 0, 1, 2],dtype=np.float32), 0, params['gaussian_filter_std'])
            ll2 = tf.nn.conv1d(tf.reshape(ll2,(-1,num_classes,1)),gf.reshape((5,1,1)),1,'SAME')
        outputs2 = tf.reduce_sum(tf.reshape(ll2,(-1,inputs_shape[1],num_classes)),axis=1,name='outputs2')
        accuracy2 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(outputs2,axis=1),tf.argmax(targets[::inputs_shape[1]],axis=1)),tf.float32),name='accuracy2')

    with tf.name_scope('train'):
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=outputs,labels=targets)
        if params['confidence_penalty']:
            h_activation = tf.nn.softmax(outputs)
            H = -tf.reduce_sum(h_activation*tf.log(h_activation),axis=1)
            error = tf.reduce_mean(loss-params['confidence_penalty']*H,name='loss')
        else:
            error = tf.reduce_mean(loss,name='loss')

        l1_lambda = params['l1_regularization']
        if l1_lambda > 0.0:
            l1_loss = l1_lambda*tf.add_n([tf.reduce_sum(tf.abs(w)) for w in fc_weights],name='l1_loss')
            error += l1_loss

        l2_lambda = params['l2_regularization']
        if l2_lambda > 0.0:
            l2_loss = 0.5*l2_lambda*tf.add_n([tf.nn.l2_loss(w) for w in fc_weights],name='l2_loss')
            error += l2_loss

        error = tf.identity(error,name='error')

        optimizer = get_optimizer(params)
        grads = optimizer.compute_gradients(error)
        #grads = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in grads]
        train = optimizer.apply_gradients(grads,name="train")

    with tf.name_scope('summaries'):
        tf.summary.scalar('error', error)
        tf.summary.scalar('accuracy', accuracy)
        # for layer in layers:
        #     tf.summary.histogram(layer.name, layer)
        # for grad, var in grads:
        #     tf.summary.histogram(var.name + '/gradient', grad)

def create_cnn_net(params, train_data):
    """Returns a tf.Graph of a convolutional neural network. dims is a list of hidden layer sizes."""
    graph = tf.Graph()
    with graph.as_default():

        keep_prob = tf.placeholder(tf.float32,name='keep_prob')

        layers = []
        create_input_layers(params,layers,train_data.inputs_shape)

        if params['dropout'][0]:
            with tf.name_scope('dropout-0'):
                layers.append(tf.nn.dropout(layers[-1],keep_prob,name='dropout'))

        num_convs = int(layers[-1].get_shape()[2])
        kdim = params['mask_size']
        for i in range(len(params['conv_dims'])):
            with tf.name_scope('conv-%d'%(i+1)):
                prev_num_convs = num_convs
                num_convs = params['conv_dims'][i]
                W, b, a = conv_layer(layers[-1],[kdim,prev_num_convs,num_convs],get_conv_weight_init(params),params['conv_padding'],get_activation(params))
                if params['gates']:
                    V, c, g = conv_layer(layers[-1],[kdim,prev_num_convs,num_convs],get_conv_weight_init(params),params['conv_padding'],get_activation(params),names=['V','c','g'])
                    layers.append(tf.multiply(a,tf.sigmoid(g),name='gate'))
                else:
                    layers.append(a)

            if params['max_pool'][i]:
                with tf.name_scope('maxpool-%d'%(i+1)):
                    layers.append(maxpool(layers[-1],params['max_pool'][i]))

            if params['dropout'][i+1]:
                with tf.name_scope('dropout-%d'%(i+1)):
                    layers.append(tf.nn.dropout(layers[-1],keep_prob,name='dropout'))

        layers.append(tf.reshape(layers[-1],[-1,int(layers[-1].get_shape()[1])*int(layers[-1].get_shape()[2])]))
        if 'mean+var' in params['features']:
            layers.append(tf.concat([layers[-1],*tf.nn.moments(tf.reshape(layers[0],[-1,train_data.window_size,25]),axes=[1])],1))

        fc_weights = create_fc_layers(params,layers)
        create_output_layer(params,layers,train_data.num_classes)
        create_eval_and_train(params,layers,train_data.inputs_shape,train_data.num_classes,fc_weights)

        summaries = tf.summary.merge_all()
        init = tf.global_variables_initializer()

    return graph, init, summaries

def create_rnn_net(params, train_data):
    """Returns a tf.Graph of a recurrent neural network. dims is a list of hidden layer sizes."""
    graph = tf.Graph()
    with graph.as_default():

        keep_prob = tf.placeholder(tf.float32,name='keep_prob')
        #input_keep_prob = tf.placeholder(tf.float32,name='input_keep_prob')
        #output_keep_prob = tf.placeholder(tf.float32,name='output_keep_prob')

        layers = []
        create_input_layers(params,layers,train_data.inputs_shape)

        with tf.name_scope('cell'):
            if len(params['rnn_dims'])>1:
                cells = []
                for i in range(len(params['rnn_dims'])):
                    cells.append(tf.contrib.rnn.DropoutWrapper(get_rnn_cell(params,i),input_keep_prob=keep_prob,output_keep_prob=keep_prob))
                cell = tf.contrib.rnn.MultiRNNCell(cells,state_is_tuple=True)
            else:
                cell = tf.contrib.rnn.DropoutWrapper(get_rnn_cell(params,0),input_keep_prob=keep_prob,output_keep_prob=keep_prob)
            val, state = tf.nn.dynamic_rnn(cell, layers[-1], dtype=tf.float32)
            val = tf.transpose(val, [1, 0, 2])
            last = tf.gather(val, int(val.get_shape()[0])-1)
            print(last.get_shape())
            layers.append(last)

        fc_weights = []
        create_output_layer(params,layers,train_data.num_classes)
        create_eval_and_train(params,layers,train_data.inputs_shape,train_data.num_classes,fc_weights)

        summaries = tf.summary.merge_all()
        init = tf.global_variables_initializer()

    return graph, init, summaries

def run_graph(params,train_data,val_data,epochs=10,logdir=None,kaggle=False):
    print('Running network with params =')
    print(sorted(params.items()))
    print()

    if params['type'] == 'cnn':
        graph, init, summaries = create_cnn_net(params, train_data)
    elif params['type'] == 'rnn':
        graph, init, summaries = create_rnn_net(params, train_data)

    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    inputs = graph.get_tensor_by_name("inputs/inputs:0")
    targets = graph.get_tensor_by_name("eval/targets:0")
    accuracy = graph.get_tensor_by_name("eval/accuracy:0")
    accuracy2 = graph.get_tensor_by_name("eval/accuracy2:0")
    outputs2 = graph.get_tensor_by_name("eval/outputs2:0")
    error = graph.get_tensor_by_name("train/error:0")
    train = graph.get_operation_by_name("train/train")

    #with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=4),graph=graph) as sess:
    with tf.Session(graph=graph) as sess:
        saver = tf.train.Saver()

        start_epoch = 0
        best_val_acc = 0.0
        stats = {'train': {'acc': [], 'err': []},
                 'val': {'acc': [], 'acc2': [], 'err': []},
                 'epochs': 0, 'runtime': []}

        if not os.path.exists(MODELDIR%logdir):
            os.makedirs(MODELDIR%logdir)
        if not os.path.exists(LOGFILE%logdir):
            print('Writing logs to',LOGFILE%logdir)
            sess.run(init)
        else:
            with open(LOGFILE%logdir,'r') as fin:
                stats = json.load(fin)['stats']
            start_epoch = len(stats['train']['err'])
            best_val_acc = stats['best_val_acc']
            print('Loaded previous logs from',LOGFILE%logdir)

            saver.restore(sess, MODELDIR%logdir+'ckpt')
            print('Restored previous model from',MODELDIR%logdir,'- resuming after %d epochs'%start_epoch)

        train_writer = tf.summary.FileWriter(MODELDIR%logdir+'train', graph)
        val_writer = tf.summary.FileWriter(MODELDIR%logdir+'val', graph)

        for epoc in range(start_epoch,epochs):
            start = time.time()
            epoc_err = 0.
            epoc_acc = 0.
            for b, (batch_inputs, batch_targets) in enumerate(train_data):
                _, batch_err, batch_acc, sum_bufs = sess.run(
                    [train,error,accuracy,summaries],
                    feed_dict={inputs:batch_inputs,targets:batch_targets,keep_prob:params['keep_prob']})
                train_writer.add_summary(sum_bufs, epoc*(train_data.num_batches+val_data.num_batches)+b)
                epoc_err += batch_err
                epoc_acc += batch_acc
            epoc_err /= train_data.num_batches
            epoc_acc /= train_data.num_batches
            print('epoch: {0}, err: {1}, acc: {2}'.format(epoc+1, epoc_err, epoc_acc))
            stats['train']['err'].append(epoc_err)
            stats['train']['acc'].append(epoc_acc)
            print('epoch train runtime: ',time.time()-start)

            val_err = 0.
            val_acc = 0.
            val_acc2 = 0.
            for b, (batch_inputs, batch_targets) in enumerate(val_data):
                batch_err, batch_acc, batch_acc2, sum_bufs = sess.run(
                    [error,accuracy,accuracy2,summaries],
                    feed_dict={inputs:batch_inputs,targets:batch_targets,keep_prob:1.0})
                val_writer.add_summary(sum_bufs, epoc*(train_data.num_batches+val_data.num_batches)+train_data.num_batches+b)
                val_err += batch_err
                val_acc += batch_acc
                val_acc2 += batch_acc2
            val_err /= val_data.num_batches
            val_acc /= val_data.num_batches
            val_acc2 /= val_data.num_batches
            print('val err: {0}, acc: {1}, acc2: {2}'.format(val_err, val_acc, val_acc2))
            stats['val']['err'].append(float(val_err))
            stats['val']['acc'].append(float(val_acc))
            stats['val']['acc2'].append(float(val_acc2))

            if best_val_acc < val_acc2:
                best_val_acc = val_acc2
                if kaggle:
                    print('Writing Kaggle output...')
                    test_inputs = np.load(os.path.join(os.environ['MLP_DATA_DIR'], 'msd-%d-genre-test-inputs.npz'%train_data.num_classes))['inputs']
                    test_inputs = test_inputs.reshape((test_inputs.shape[0], -1))
                    test_pred_classes = np.empty(test_inputs.shape[0])
                    for i in range(0,test_inputs.shape[0],val_data.batch_size):
                        test_pred_classes[i:i+val_data.batch_size] = sess.run(tf.nn.softmax(outputs2), feed_dict={inputs: val_data.create_sliding_windows(test_inputs[i:i+val_data.batch_size]),keep_prob:1.0}).argmax(-1)
                    create_kaggle_submission_file(test_pred_classes, MODELDIR%logdir+'submission-%.2f.csv'%(best_val_acc*100), True)

            stats['epochs'] = epoc+1
            stats['best_val_acc'] = best_val_acc
            runtime = time.time()-start
            stats['runtime'].append(runtime)
            print('runtime: ',runtime)
            sys.stdout.flush()

            saver.save(sess, MODELDIR%logdir+'ckpt')
            with open(LOGFILE%logdir,'w') as fout:
                json.dump({'params':params, 'stats': stats}, fout)

        #cm = confusion_matrix(val_data.targets,np.argmax(outputs2.eval(feed_dict={inputs:val_data.inputs.reshape((-1,val_data.inputs.shape[1]/split,train_data.num_classes)),dropout_input:1.0,droput_output:1.0}),axis=1))
        #stats['confusion_matrix'] = cm.tolist()

    return stats


if __name__ == '__main__':

    logdir = sys.argv[1]
    num_epochs = int(sys.argv[2])

    with open(PARAMFILE%logdir,'r') as f:
        params = json.load(f)

    train_data = MSDGenreDataProvider(params.get('train','train'), batch_size=params['batch_size'], num_classes=params['classes'], window_size=params['window_size'], shuffle_order=False,stride=params['stride'], disturb_label=params['disturb_label'])
    print(train_data.inputs.shape, train_data.targets.shape, train_data.inputs_shape)
    val_data = MSDGenreDataProvider('valid', batch_size=params['batch_size'], num_classes=params['classes'], window_size=params['window_size'], stride=params['stride'])
    print(val_data.inputs.shape, val_data.targets.shape, val_data.inputs_shape)

    stats = run_graph(params,train_data,val_data,epochs=num_epochs,logdir=logdir,kaggle=True)
    print('Finished training, best val acc', stats['best_val_acc'])
