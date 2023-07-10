import gym
import random
import os
import numpy as np
from collections      import deque
from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam

class Agent():
    def __init__(self, state_size, action_size):
        self.weight_backup      = "cartpole_weight.h5"
        self.state_size         = state_size
        #用来存储state，在本项目中，state有四种位置，速度，杆的角度，角速度。所以size是4
        self.action_size        = action_size
        #用来存储action，在本项目中，action只有两种状态，左和右。所以size是2
        self.memory             = deque(maxlen=2000)
        #双端队列，最大长度为2000，会保存将state，action，reward，next_state，done
        self.learning_rate      = 0.001
        #学习速度
        self.gamma              = 0.95
        #折扣因子，0.95表明这个算法更加注重于对于未来奖励的回报
        self.exploration_rate   = 1.0
        #采取随机探索的概率
        self.exploration_min    = 0.01
        #最小随机探索的概率
        self.exploration_decay  = 0.995
        #探索率每次更新时的衰减因子
        self.brain              = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        #设置了一个sequential模型，作为神经网络
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        #建立一个sequential模型，形成多个网络层的线性堆叠。同时添加两个全连接层，每层都有24个神经元
        #在第一个Dense层，输入指定数据的维度，同时在两者中都适用relu作为激活函数
        #最后第三个函数，使用的维度是action的维度（2），用线性函数来激活。主要是为了输出每个动作的Q值
        #最后用compile方法来编译模型，损失函数为均方误差，优化器为Adam

        if os.path.isfile(self.weight_backup):
            model.load_weights(self.weight_backup)
            self.exploration_rate = self.exploration_min
        return model
        #查询是否有已经训练好的模型文件，如果有的话，就把探索值改为最小，最大倾向于使用已有的方法

    def save_model(self):
            self.brain.save(self.weight_backup)
            #用来保存数据

    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        act_values = self.brain.predict(state)
        return np.argmax(act_values[0])
        #本函数是用来采取动作的，产生一个随机数，如果随机数小于或者等于当前的探索率，那就采取一个随机的动作，否则的话，使用模型来预测，并且采用Q值最大的动作
        #这个是基于ε-greedy policy

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        #储存经验，基于强化学习的经验回放机制

    def replay(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            return
        sample_batch = random.sample(self.memory, sample_batch_size)
        #如果当前记忆库的大小小于设定的批次大小，则不进行训练，直接返回
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            #初始化的目标值就是奖励
            if not done:
              target = reward + self.gamma * np.amax(self.brain.predict(next_state)[0])
              #Q-learning 算法
            target_f = self.brain.predict(state)
            #计算当前state所有action的预测值
            target_f[0][action] = target
            #对于我们实际执行的动作值，我们希望其接近我们的之前计算的目标值，所以将其赋值为目标值
            self.brain.fit(state, target_f, epochs=1, verbose=0)
            #使用新的目标值和原始状态在进行一次训练
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay
            #随着训练进行，不断的降低探索率

class CartPole:
    def __init__(self):
        self.sample_batch_size = 32
        self.episodes          = 10000
        self.env               = gym.make('CartPole-v1')
        #每次从记忆库里抽取32个样本训练，一共训练2000回，训练环境为cartpole

        self.state_size        = self.env.observation_space.shape[0]
        print("State size:", self.state_size)  # 这行代码会打印状态空间的大小
        self.action_size       = self.env.action_space.n
        self.agent             = Agent(self.state_size, self.action_size)
        #获取环境的空间大小，获取可以采取行动的大小，创建一个智能体，并传递状态大小和行动大小


    def run(self):
        try:
            for index_episode in range(self.episodes):
                # state = self.env.reset()
                state = self.env.reset()  # 修改这一行
                #每次开始时，将状态重塑
                print("Initial state:", state)  # 添加这行代码以打印初始状态
                state = np.reshape(state, [1, self.state_size])
                #将状态重塑，使其可以作为智能体的神经网络输入

                done = False
                index = 0
                while not done:
                    # self.env.render()

                    action = self.agent.act(state)
                    #根据当前状态，智能体选择一个动作
                    print(self.env.step(action))  # 添加这行代码以打印 step 函数的返回值
                    # next_state, reward, done, _ = self.env.step(action)
                    next_state, reward, done, _ = self.env.step(action)  # 添加一个额外的变量来接收第五个返回值
                    #获取返回值

                    print("Next state:", next_state)  # 添加这行代码以打印每一步之后的状态
                    next_state = np.reshape(next_state, [1, self.state_size])
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    #重塑并记忆返回值们，然后把现在的状态变为下一个状态
                    index += 1
                print("Episode {}# Score: {}".format(index_episode, index + 1))
                self.agent.replay(self.sample_batch_size)
                #在记忆库中随机收取一批数据进行训练
        finally:
            self.agent.save_model()
            #保存这个模型

if __name__ == "__main__":
    cartpole = CartPole()
    cartpole.run()
