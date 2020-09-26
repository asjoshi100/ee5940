import torch as pt
import numpy as np
import functools as ft

crossMat = lambda omega : pt.cross(-omega.repeat((3,1)),
                                   pt.eye(3,dtype=omega.dtype),dim=1) 


def rodrigues(v,theta):
    # Assumes that 
    # v is a unit vector
    v_hat = crossMat(v)
    
    return pt.eye(3) + pt.sin(theta)*v_hat + (1-pt.cos(theta))*(v_hat@v_hat)

def toSE3(R,p):
    M_top = pt.cat([R,p.view((3,1))],dim=1)
    M_bot = pt.cat([pt.zeros((1,3),dtype=R.dtype),
                    pt.ones((1,1),dtype=R.dtype)],dim=1)
    M = pt.cat([M_top,
                M_bot],dim=0)
    return M
def rotationSE3(v,theta):
    #Omega = crossMat(omega)
    R = rodrigues(v,theta)
    p = pt.zeros(3,dtype=v.dtype)
    
    return toSE3(R,p)



def translateSE3(p):
    R = pt.eye(3,dtype=p.dtype)
    return toSE3(R,p)

Rx = lambda q : rotationSE3(pt.tensor([1,0,0.],dtype=q.dtype),q)
Rz = lambda q : rotationSE3(pt.tensor([0.,0,1],dtype=q.dtype),q)

Tx = lambda q : translateSE3(pt.tensor([q,0,0]))
Tz = lambda q : translateSE3(pt.tensor([0,0.,q]))

def DHMat(theta,d,a,alpha):
    return Rz(theta)@Tz(d)@Tx(a)@Rx(alpha)

def Revolute(d=0,a=0,alpha=0,offset=pt.tensor(0.)):
    def jointMat(theta):
        d_ten = d.clone().detach().type(theta.dtype)
        a_ten = a.clone().detach().type(theta.dtype)
        alpha_ten = alpha.clone().detach().type(theta.dtype)
        offset_ten = offset.clone().detach().type(theta.dtype)
        return DHMat(theta+offset_ten,d_ten,a_ten,alpha_ten)
    return jointMat

class SerialLink:
    def __init__(self,ListOfLinks,base=None,tool=None):
        self.Links = ListOfLinks
        self.base = base
        self.tool = tool
        
    def fkine(self,q):
        M_list = [L(qi) for L,qi in zip(self.Links,q)]
        if self.tool is not None:
            M_list.append(self.tool)
        if self.base is not None:
            M_list = [self.base] + M_list
        return ft.reduce(pt.mm,M_list,pt.eye(4,dtype=q.dtype))


class PUMA(SerialLink):
    """
    Parameters from
    
    https://github.com/petercorke/robotics-toolbox-matlab/blob/master/models/mdl_puma560.m
    
    More info at
    https://github.com/petercorke/robotics-toolbox-matlab
    """
    def __init__(self,base=None,tool=None):
        # DH Parameters
        self.d = pt.tensor([0,0,.15,.4318,0,0])
        self.a = pt.tensor([0,.4318,.0203,0,0,0])
        self.alpha = (np.pi/2) * pt.tensor([1,0,-1,1,-1,0])
        
        
        
        Links = [Revolute(d,a,alpha) for d,a,alpha in \
                zip(self.d,self.a,self.alpha)]
        super().__init__(Links,base,tool)
        
        
        # Link Masses
        self.m = pt.tensor([0.,17.4,4.8,0.82,0.34,0.09])
        
        # Link inertias
        # Stored as Ixx,Iyy,Izz,Ixy,Iyz,Ixz
        self.I = pt.tensor([[0,.35,0,0,0,0],
                           [0.13, 0.524, 0.539, 0, 0, 0],
                           [0.066, 0.086, 0.0125, 0, 0, 0],
                           [1.8e-3, 1.3e-3, 1.8e-3, 0, 0, 0],
                           [0.3e-3, 0.4e-3, 0.3e-3, 0, 0, 0],
                           [0.15e-3, 0.15e-3, 0.04e-3, 0, 0, 0]])
        
        # Only works because they are diagonal inertias
        self.I = pt.stack([pt.diag(I[:3]) for I in self.I],dim=0)
        
        # Location of link center of mass
        self.r = pt.tensor([[0, 0, 0],
                           [-0.3638, 0.006, 0.2275],
                           [-0.0203, -0.0141, 0.070],
                           [0, 0.019, 0],
                           [0, 0, 0],
                           [0, 0, 0.032]])
        
        # Gear ratios
        self.G = pt.tensor([-62.6111,107.815,-53.7063,76.0364,71.923,76.686])
        
        # Actuator inertias
        self.Jm = pt.tensor([200e-6,200e-6,200e-6,33e-6,33e-6,33e-6])
        
        # Viscous Friction of actuator
        self.B = pt.tensor([1.48e-3,.817e-3,1.38e-3,71.2e-6,82.6e-6,36.7e-6])
        
        # Coulomb friction at actuator
        self.Tc = pt.tensor([[0.395, -0.435],
                            [0.126, -0.071],
                            [0.132, -0.105],
                            [11.2e-3, -16.9e-3],
                            [9.26e-3, -14.5e-3],
                            [3.96e-3, -10.5e-3]])
        
        deg = np.pi/180.
        # Joint Limits
        self.qlim = deg * pt.tensor([[-160, 160.],
                                    [-45, 225],
                                    [-225, 45],
                                    [-110, 170],
                                    [-100, 100],
                                    [-266, 266]])
    def ikine(self,M):
        px,py,pz = M[:3,-1]
       
        p = M[:3,-1]
        
        a2 = self.a[1]
        a3 = self.a[2]
        d3 = self.d[2]
        d4 = self.d[3]
        
        ax = M[0,2]
        ay = M[1,2]
        az = M[2,2]
        
        ox,oy,oz = M[:3,1]
  
        # First angle
        r = pt.norm(p[:2])
        phi = pt.atan2(py,px)
        t_int = pt.asin(d3/r)
        Theta_1 = [phi+t_int,phi-t_int+np.pi]
        
        # Second angle
        
        Thetas = []
        for theta1 in Theta_1:
            c1 = pt.cos(theta1)
            s1 = pt.sin(theta1)
            
            V114 = c1 * px + s1 * py
            r = pt.sqrt(V114**2 + pz**2)
            
            num = a2**2 - d4**2 - a3**2 + V114**2 + pz**2
            den = 2*a2*r
            psi = pt.acos(num/den)
            phi = pt.atan2(pz,V114)
            
            Theta_2 = [phi+psi,phi-psi]
            for theta2 in Theta_2:
                # theta3
                c2 = pt.cos(theta2)
                s2 = pt.sin(theta2)
                phi = pt.atan2(a3,d4)
                num = c2 * V114+s2*pz - a2
                den = c2*pz - s2 * V114
                psi = pt.atan2(num,den)
                theta3 = phi-psi
                
                # theta4
                c23 = pt.cos(theta2+theta3)
                s23 = pt.sin(theta2+theta3)
                V323 = c1 * ay - s1 * ax
                V113 = c1 * ax + s1 * ay
                V313 = c23 * V113 + s23 * az
                
                Theta_4 = [pt.atan2(-V323,-V313),
                           pt.atan2(V323,V313)]
                for theta4 in Theta_4:
                    # theta5
                    c4 = pt.cos(theta4)
                    s4 = pt.sin(theta4)
                    s5 = -c4 * V313 - s4 * V323
                    c5 = -s23 * V113 + c23 * az
                    theta5 = pt.atan2(s5,c5)
                    
                    # theta6
                    V132 = s1 * ox - c1 * oy
                    V112 = c1 * ox + s1 * oy
                    V332 = -s23 * V112 + c23 * oz
                    V312 = c23 * V112 + s23 * oz
                    V432 = s4 * V312 + c4 * V132
                    V422 = V332
                    V412 = c4 * V312 - s4 * V132
                    
                    s6 = -c5 * V412 - s5 * V422
                    c6 = -V432
                    theta6 = pt.atan2(s6,c6)
                    Thetas.append([theta1,
                                   theta2,
                                   theta3,
                                   theta4,
                                   theta5,
                                   theta6])
                    
        return pt.tensor(Thetas)
    
    def rne(self,q,q_dot,q_ddot,gravity=None,
            fext=None):
        
        """
        Based on 
        
        Luh, John YS, Michael W. Walker, and Richard PC Paul. 
        "On-line computational scheme for mechanical manipulators." (1980): 69-76.
        
        and the corresponding implementation from
        https://github.com/petercorke/robotics-toolbox-matlab/blob/master/%40SerialLink/rne_dh.m
        
        
        """
        
        if gravity is None:
            gravity = pt.tensor([0,0,9.81],dtype=q.dtype)
        if fext is None:
            fext = pt.zeros(6,dtype=q.dtype)
            
        n = len(q)
        z0 = pt.tensor([0,0,1.],dtype=q.dtype)
        
            
        
        
        # Forward Pass
        Rb = pt.eye(3,dtype=q.dtype)
        w = pt.zeros(3,dtype=q.dtype)
        wd = pt.zeros(3,dtype=q.dtype)
        vd = gravity.clone()
           
        Forces = []
        Torques = []
        
        Pstar = []
        Rotations = []
        for i in range(len(q)):
            qi = q[i]
            qdi = q_dot[i]
            qddi = q_ddot[i]
            r = self.r[i].clone().detach().type(q.dtype)
            
            T = self.Links[i](qi)
            R = T[:3,:3]
            Rotations.append(R)
            Rt = R.T
            
            d = self.d[i]
            alpha = self.alpha[i]
            pstar = pt.tensor([self.a[i],d*pt.sin(alpha),d*pt.cos(alpha)]).type(q.dtype)
            Pstar.append(pstar)
            
            wd = Rt@(wd + z0*qddi + pt.cross(w,z0*qdi) )
            w = Rt@(w + z0*qdi)
            vd = pt.cross(wd,pstar) + pt.cross(w,pt.cross(w,pstar)) + Rt@vd
            
            vhat = pt.cross(wd,r) + pt.cross(w,pt.cross(w,r)) + vd
            F = self.m[i] * vhat
            I = self.I[i].clone().detach().type(q.dtype)
            N = I @ wd + pt.cross(w,I@w)
            
            Forces.append(F)
            Torques.append(N)
        # Backward pass
        
        f = fext[:3]
        nn = fext[3:]
        
        jointTorques = []
        for i in range(n)[::-1]:
            pstar = Pstar[i]
            
            if i == n-1:
                R = pt.eye(3,dtype=q.dtype)
            else:
                R = Rotations[i+1]
                
            
            r = self.r[i].clone().detach().type(q.dtype)
            

            nn = R@(nn+pt.cross(R.T@pstar,f)) + pt.cross(pstar+r,Forces[i]) + Torques[i]
            f = R@f + Forces[i]
            
            
            
            R = Rotations[i]
            
            G = self.G[i]
            Jm = self.Jm[i]
            
            # friction force
            B = self.B[i]
            Tc = self.Tc[i]
            tau = B * pt.abs(G) * q_dot[i]
            
            # This is coulomb friction
            # It causes lots of problems in simulation
            #if q_dot[i] > 0:
            #    tau = tau + Tc[0]
            #elif q_dot[i] < 0:
            #    tau = tau + Tc[1]
                
            tau = -pt.abs(G) * tau

            # Total torque
            t = nn@R.T@z0 + G**2 * Jm * q_ddot[i] - tau
            jointTorques.append(t)
        return pt.stack(jointTorques[::-1])
    def MassMatrix(self,q):
        MM = []
        for qdd in pt.eye(len(q),dtype=q.dtype):
            tau = self.rne(q,pt.zeros_like(q),qdd,gravity=pt.zeros(3,dtype=q.dtype))
            MM.append(tau)
            
        return pt.stack(MM)
    
    def accel(self,q,q_dot,tau):
        M = self.MassMatrix(q)
        tau_grav = self.rne(q,q_dot,pt.zeros_like(q))
        
        R = tau-tau_grav
        q_ddot,_ = pt.solve(R.view((len(q),1)),M)
        return q_ddot.view((len(q),))
        
    def vectorField(self,x,tau):
        q = x[:6]
        q_dot = x[6:]
        q_ddot = self.accel(q,q_dot,tau)
        return pt.cat([q_dot,q_ddot])

    def grav_torque(self,q):
        return self.rne(q,pt.zeros_like(q),pt.zeros_like(q))

