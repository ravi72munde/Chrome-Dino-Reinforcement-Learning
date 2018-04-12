import numpy as np
import io
import time
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (15, 9)
import seaborn as sns
import pandas as pd
import numpy as np
from IPython.display import clear_output
from random import randint
import game as Game
import dinoAgent as DinoAgent

def trainNetwork(model,game_state,observe=False):
    # open up a game state to communicate with emulator
    last_time = time.time()
    # store the previous observations in replay memory
    D = load_obj("D") #deque()
    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] =1 
    x_t, r_0, terminal = game_state.get_state(do_nothing)
    

    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    

    #In Keras, need to reshape
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  #1*80*80*4
    initial_state = s_t

    if observe :#args['mode'] == 'Run':
        OBSERVE = 999999999    #We keep observe, never train
        epsilon = FINAL_EPSILON
        print ("Now we load weight")
        model.load_weights("model.h5")
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse',optimizer=adam)
        print ("Weight load successfully")    
    else:                       #We go to training mode
        OBSERVE = OBSERVATION
        epsilon = load_obj("epsilon")#FINAL_EPSILON #INITIAL_EPSILON
        model.load_weights("model.h5")
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse',optimizer=adam)

    t = load_obj("time")
    while (True):
        
        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0
        a_t = np.zeros([ACTIONS])
        #choose an action epsilon greedy
        if t % FRAME_PER_ACTION == 0:
            if  random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                q = model.predict(s_t)       #input a stack of 4 images, get the prediction
                max_Q = np.argmax(q)
                action_index = max_Q
                a_t[action_index] = 1
        #We reduced the epsilon gradually
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE #update to original asap

        #run the selected action and observed next state and reward
        x_t1, r_t, terminal = game_state.get_state(a_t)
        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1) #1x80x80x1
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        # store the transition in D
        D.append((s_t, action_index, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        #only train if done observing
        if t > OBSERVE: 
            
            #sample a minibatch to train on
            minibatch = random.sample(D, BATCH)
            inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))   #32, 20, 40, 4
            targets = np.zeros((inputs.shape[0], ACTIONS))                         #32, 2

            #Now we do the experience replay
            for i in range(0, len(minibatch)):
                state_t = minibatch[i][0]
                action_t = minibatch[i][1]   #This is action index
                reward_t = minibatch[i][2]
                state_t1 = minibatch[i][3]
                terminal = minibatch[i][4]
                # if terminated, only equals reward

                inputs[i:i + 1] = state_t    #I saved down s_t

                targets[i] = model.predict(state_t)  # Hitting each buttom probability
                Q_sa = model.predict(state_t1)
                
                if terminal:
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)
            # targets2 = normalize(targets)
            loss += model.train_on_batch(inputs, targets)
            loss_df.loc[len(loss_df)] = loss
        else:
            #artificial time delay as training done with this delay
            loss_df.loc[len(loss_df)] = 0
            time.sleep(0.15)
        s_t = initial_state if terminal else s_t1
        t = t + 1
        
        # save progress every 10000 iterations
        if t % 1000 == 0:
            print("Now we save model")
            
            model.save_weights("model.h5", overwrite=True)
            save_obj(D,"D") #saving episodes
            save_obj(t,"time") #caching time steps
            save_obj(epsilon,"epsilon") #cache epsilon to avoid repeated randomness in actions
            loss_df.to_csv("./objects/loss_df.csv",index=False)
            scores_df.to_csv("./objects/scores_df.csv",index=False)
            actions_df.to_csv("./objects/actions_df.csv",index=False)
            clear_output()
            with open("model.json", "w") as outfile:
                json.dump(model.to_json(), outfile)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state,             "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t,             "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)

    print("Episode finished!")
    print("************************")


def playGame(observe=False):
    game = Game()
    dino = DinoAgent(game)
    game_state = Game_sate(dino,game)
    model = buildmodel()
    trainNetwork(model,game_state,observe=observe)

def main():
	loss_df = pd.read_csv("./objects/loss_df.csv")
	scores_df = pd.read_csv("./objects/scores_df.csv")
	actions_df = pd.read_csv("./objects/actions_df.csv")
	q_max_df = pd.read_csv("./objects/q_values.csv")
	#game parameters
	ACTIONS = 2 # possible actions: jump, do nothing
	GAMMA = 0.8 # decay rate of past observations original 0.99
	OBSERVATION = 20000. # timesteps to observe before training
	EXPLORE = 50000 #300000. # frames over which to anneal epsilon
	FINAL_EPSILON = 0.0001 # final value of epsilon
	INITIAL_EPSILON = 0.1 # starting value of epsilon
	REPLAY_MEMORY = 50000 # number of previous transitions to remember
	BATCH = 32 # size of minibatch
	FRAME_PER_ACTION = 4
	LEARNING_RATE = 1e-4

	"""initial variable caching, done only once"""
	# save_obj(FINAL_EPSILON,"epsilon")
	# t = 0
	# save_obj(t,"time")
	# D = deque()
	# save_obj(D,"D")

	img_rows , img_cols = 40,20
	img_channels = 4 #We stack 4 frames
	playGame(observe = False)








