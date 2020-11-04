import numpy as np
import numpy.random as rnd
import casadi as cdi
import torch as pt
import scipy.integrate as itg
import gym



#### Parameters ####
M = 1.
m = 1.
ell = 1.
g = 9.8
dt = 0.1

#### State #####
X = cdi.SX.sym('X',4)

p = X[0]
theta = X[1]
v = X[2]
omega = X[3]


#### Input #####
U = cdi.SX.sym('U')

#### Dynamics ####
det = m*ell**2 * (M+m*cdi.sin(theta)**2)

v_dot = (m**2 * ell** 3 * omega**2 * cdi.sin(theta) - \
         m**2 * ell**2 * g * cdi.sin(theta) * cdi.cos(theta) + \
         m * ell**2 * U) / det

omega_dot = (-m**2 * ell**2 * omega**2 *cdi.sin(theta)*cdi.cos(theta) + \
            (M+m)*m*g*ell*cdi.sin(theta) - m*ell * cdi.cos(theta) * U) / det

X_dot = cdi.vcat([v,omega,v_dot,omega_dot])

##### Integration Via Scipy #####
vectorFieldFun = cdi.Function('f',[X,U],[X_dot])

##### The Cost Function #####
cost = 1-cdi.cos(theta) + p**2
# The cost penalizes the deviation from vertical
# As well as the distance from 0
costFun = cdi.Function('cost',[X,U],[cost])

##### A Gym-like class ########
U_max = 10
U_min = -10
class Pendulum:
    def __init__(self):
        self.UB_U = np.array([U_max])
        self.LB_U = np.array([U_min])
        self.UB_X = np.array([2.,20,10,20])
        self.LB_X = -self.UB_X

        self.action_space = gym.spaces.Box(self.LB_U,self.UB_U,dtype=self.UB_U.dtype)
        self.observation_space= gym.spaces.Box(self.LB_X,self.UB_X,dtype=self.UB_X.dtype)
        
        self.x0 = np.array([0,np.pi,0,0])

        self.dt = dt
        
        self.maxCost = self.cost(np.array([self.UB_X[0],np.pi,0,0]),np.zeros(1))
    def isFeasible(self,x,u):
        x_feas = np.min(x>=self.LB_X) and np.min(x <= self.UB_X)
        u_feas = np.min(u>=self.LB_U) and np.min(u <= self.UB_U)
        return x_feas and u_feas
        
    def cost(self,x,u):
        p,theta = x[:2]
        return 1-np.cos(theta)+p**2
    
    def vectorField(self,t,x,u):
        x_dot = vectorFieldFun(x,u)
        return np.array(x_dot).squeeze()
    
    def step_from_state(self,x,u):
        done = not self.isFeasible(x,u)
        c = self.cost(x,u)
        info = {}
        if done:
            # If infeasible
            # Stay exactly where you are
            c = self.maxCost
            return x,c,done,info

        res = itg.solve_ivp(self.vectorField,(0,dt),x,args=(u,))
        x_next = res.y[:,-1]

        done = not self.isFeasible(x_next,u)
        if done:
            c = self.maxCost
        return x_next,c,done,info
    
    def step(self,u):
        x_next,c,done,info = self.step_from_state(self.x,u)
        self.x = np.copy(x_next)
        return x_next,c,done,info

    def reset(self):
        self.x = self.x0 + .1 * self.observation_space.sample()
        return np.copy(self.x)
        
##### Passivity-Based Controller #######

# Swing-Up controller Parameters
   
class PassivitySwingUp:
    """
    A version of the swing up controller which has
    been clipped for feasibility
    """
    def __init__(self,Kv = 4 * m * g * ell + 1.,
                 Kx = .01,
                 Kdec = 100.):

        self.Kv = Kv
        self.Kx = Kx
        self.Kdec = Kdec
        q_dot = X[2:]
    
    
        Mass = cdi.vcat([cdi.hcat([M+m,m*ell*cdi.cos(theta)]),
                         cdi.hcat([m*ell*cdi.cos(theta),m*ell**2])])
    
        KE = .5 * q_dot.T @ Mass @ q_dot
        PE = m*g*ell * (cdi.cos(theta)-1)
        E = KE + PE
    
        scaledDet = (M+m*cdi.sin(theta)**2)
        den = E + Kv / scaledDet
    
        num = -Kdec * v - Kx * p + \
            (g*cdi.cos(theta) - ell*omega**2) * \
            Kv *m*cdi.sin(theta)/scaledDet
    
        U_swingUp = num/den

        #### Functional Form #####
        self.swingUpFun = cdi.Function('swingUp',[X],[U_swingUp])

       

    def action(self,x):
        u_cdi = self.swingUpFun(x)
        u_np = np.array(u_cdi).reshape((1,))
        return np.clip(u_np,U_min,U_max)

    def update(self,x,u,c,x_next,done,info):
        pass

#### MPC ######

class MPCAgent:
    def __init__(self,N):
        #### Optimization Object #####
        opti = cdi.Opti()
        U_traj = opti.variable(N)
        x0_param = opti.parameter(4)

        
        ##### Auxiliary Dynamics ####
        zeta = cdi.SX.sym('zeta')
        Y = cdi.vcat([zeta,X])
        Y_dot = cdi.vcat([cost,X_dot])

        # Now the ODE takes U as a parameter  
        # This make the simulation outputs differentiable with respect to U
        ode = {'x' : Y, 'ode' : Y_dot, 'p' : U}
        # If the optimization is very slow, you can switch 'cvodes' to 'rk' for quicker 
        # but less accurate solutions
        opts = {'tf' : dt}  
        Sim = cdi.integrator('Sim','rk',ode,opts)

        ##### Build the simulation with variable inputs #####
        x_prob = x0_param
        y_prob = cdi.vcat([0,x_prob])
        for i in range(N):
            u_prob = U_traj[i]
            y_prob = Sim(x0=y_prob,p=u_prob)['xf']
        
        ##### Set objective ######
        opti.minimize(y_prob[0])
    
        #### Set constraints #####    
        opti.subject_to(opti.bounded(U_min,U_traj,U_max))

        # Setting a very loose tolerance for faster solves
        opti.solver('ipopt',{'verbose' : False,'print_time' :False},
                    {'tol' : 1e-3,'print_level' : 0,'warm_start_init_point' : "yes"})
        # Saving these are part of the object for later calling
        self.U_traj = U_traj
        self.opti = opti
        self.x0_param = x0_param

        self.U_init = cdi.DM.zeros(N)
    def action(self,x):
        self.opti.set_value(self.x0_param,x)
        sol = self.opti.solve()
        U_val = sol.value(self.U_traj)
        u = np.array([U_val[0]])
        return u
        
    def update(self,x,u,c,x_next,done,info):
        pass
        
