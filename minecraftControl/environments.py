import gym
import minecraftControl.minecraft as mc
import minecraftControl.controllers as ctrl
import numpy as np
import numpy.random as rnd
import minecraftControl.vehicles as vh

class constrainedBall(gym.Env):
    def __init__(self):
        pos = np.zeros(2)
        vel = np.zeros(2)
        vehicle = vh.rollingSphere(pos,vel,.4,mc.VEHICLE_SPEED,controller=None)
        self.model = mc.Model(vehicle,mc.smallLayout) 
        self.vehicle = vehicle
        self.window = None
        self.dt = .05

        self.action_space = gym.spaces.Box(low=-1.,high=1.,shape=(2,),dtype=np.float32)
        xHigh = np.array([2.5,2.5,.4,.4])
        self.observation_space = gym.spaces.Box(low=-xHigh,high=xHigh,dtype = np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset_info(self):
        self.Done = False
        self.info = { 'InputFeasible' : True,
                      'StateFeasible' : True }
        
    def initialize_position(self):
        
        pos = 5 * (rnd.rand(2) - .5)
        while np.max(np.abs(pos)) < 1.5:
            pos = 5 * (rnd.rand(2) - .5)
        vel = np.zeros(2)
        self.x0 = np.hstack([pos,vel])
 
    def make_window(self):
        window = mc.Window(model=self.model,position=(0,3,0),flying=True,
                           height=800,width=800, caption='Pyglet',
                           resizable=True)


        self.window = window

    def step(self,action):

        if not self.action_space.contains(action):
            self.Done = True
            self.info['InputFeasible'] = False

        measurement = self.get_measurement()
        if not self.observation_space.contains(measurement):
            self.Done = True
            self.info['StateFeasible'] = False
            
        if not self.Done:
            # Check the input
            self.model.vehicle.update(self.dt,action)
        measurement = self.get_measurement()
        reward = self.model.vehicle.get_reward()

        measurement = self.get_measurement()
        if not self.observation_space.contains(measurement):
            self.Done = True
            self.info['StateFeasible'] = False
        
        return measurement,reward,self.Done,self.info
        

    def get_measurement(self):
        return self.model.vehicle.x
    
    def render(self):
        if self.window is None:
            self.make_window()
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        self.window.dispatch_event('on_draw')
        self.window.on_draw()
        self.window.flip()

    def close(self):
        if self.window is not None:
            self.window.close()
            self.window = None


    def reset(self):
        self.initialize_position()
        self.model.vehicle.set_state(self.x0)
        self.reset_info()
        return self.get_measurement()
