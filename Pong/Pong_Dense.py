import tensorflow as tf
import gym
import numpy as np

gamma = 0.99
learning_rate = 0.001
decay=0.99
save_path='models/pong.ckpt'
render = True
n_actions = 3

env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None
xs,rs,ys = [],[],[]
running_reward = None
reward_sum = 0
episode_number = 0

def preprocess(input):
	input = input[35:195]
	input = input[::2,::2,0]
	input[input == 144] = 0
	input[input == 109] = 0
	input[input != 0] = 1

	return input.astype(np.float).ravel()

def get_obs(input, prev_input):
	if prev_input is not None:
		return (input-prev_input) 
	else:
		return np.zeros_like(input)

def discount_rewards(tf_r):
	discount_f = lambda a,v: a*gamma+v
	tf_r_reverse = tf.scan(discount_f,tf.reverse(tf_r,[0]))
	tf_discounted_r = tf.reverse(tf_r_reverse,[0])
	return tf_discounted_r

def feed_forward(input):
	h2 = tf.layers.dense(input,256,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
	h3 = tf.layers.dense(h2,n_actions,activation=tf.nn.softmax,kernel_initializer=tf.contrib.layers.xavier_initializer())
	return h3

with tf.device("/cpu:0"):
	tf_x = tf.placeholder(dtype=tf.float32, shape=[None,6400],name="tf_x")
	tf_y = tf.placeholder(dtype=tf.float32,shape=[None,n_actions],name="tf_y")
	tf_epr = tf.placeholder(dtype=tf.float32,shape=[None,1], name="tf_epr")

	tf_discounted_epr = discount_rewards(tf_epr)
	tf_mean, tf_variance= tf.nn.moments(tf_discounted_epr, [0], shift=None, name="reward_moments")
	tf_discounted_epr -= tf_mean
	tf_discounted_epr /= tf.sqrt(tf_variance + 1e-6)

tf_prob = feed_forward(tf_x)
loss = tf.nn.l2_loss(tf_y-tf_prob)

optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,decay=decay)
tf_grads = optimizer.compute_gradients(loss,var_list=tf.trainable_variables(), grad_loss=tf_discounted_epr)
train_op = optimizer.apply_gradients(tf_grads)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# try load saved model
saver = tf.train.Saver(tf.global_variables())
load_was_success = True # yes, I'm being optimistic
try:
    save_dir = '/'.join(save_path.split('/')[:-1])
    ckpt = tf.train.get_checkpoint_state(save_dir)
    load_path = ckpt.model_checkpoint_path
    saver.restore(sess, load_path)
except:
    print "no saved model to load. starting new session"
    load_was_success = False
else:
    print "loaded model: {}".format(load_path)
    saver = tf.train.Saver(tf.global_variables())
    episode_number = int(load_path.split('-')[-1])

# training loop
while True:
    if render:
    	env.render()

    # preprocess the observation, set input to network to be difference image
    cur_x = preprocess(observation)
    x = get_obs(cur_x,prev_x)
    prev_x = cur_x

    # stochastically sample a policy from the network
    feed = {tf_x: np.reshape(x, (1,-1))}
    prob = sess.run(tf_prob,feed)[0,:]
    action = np.random.choice(n_actions,p=prob)
    label = np.zeros_like(prob)
    label[action] = 1

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action+1)
    reward_sum += reward
    
    # record game history
    xs.append(x) ; ys.append(label) ; rs.append(reward)
    
    if done:
        # update running reward
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        
        # parameter update
       	feed = {tf_x: xs, tf_epr: np.vstack(rs), tf_y: np.vstack(ys)}
        _ = sess.run(train_op,feed)
        
        # print progress console
        if episode_number % 10 == 0:
            print 'ep {}: reward: {}, mean reward: {:3f}'.format(episode_number, reward_sum, running_reward)
        else:
            print '\tep {}: reward: {}'.format(episode_number, reward_sum)
        
        # bookkeeping
        xs,rs,ys = [],[],[] # reset game history
        episode_number += 1 # the Next Episode
        observation = env.reset() # reset env
        reward_sum = 0
        if episode_number % 50 == 0:
            saver.save(sess, save_path, global_step=episode_number)
            print "SAVED MODEL #{}".format(episode_number)