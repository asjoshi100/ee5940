import numpy as np
import scipy.linalg as la
import functools as ft

crossMat = lambda omega : np.column_stack([np.cross(omega,e) for e in np.eye(3)])

def rotationSE3(omega):
    Omega = crossMat(omega)
    R = la.expm(Omega)
    M = np.block([[R,np.zeros((3,1))],
                  [np.zeros((1,3)),np.ones((1,1))]])
    return M

def translateSE3(p):
    M = np.block([[np.eye(3),p.reshape((3,1))],
                  [np.zeros((1,3)),np.ones((1,1))]])
    return M

Rx = lambda q : rotationSE3(np.array([q,0,0.]))
Rz = lambda q : rotationSE3(np.array([0.,0,q]))

Tx = lambda q : translateSE3(np.array([q,0,0]))
Tz = lambda q : translateSE3(np.array([0,0.,q]))

def DHMat(theta,d,a,alpha):
    return Rz(theta)@Tz(d)@Tx(a)@Rx(alpha)

def Revolute(d=0,a=0,alpha=0,offset=0.):
    def jointMat(theta):
        return DHMat(theta+offset,d,a,alpha)
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
        return ft.reduce(np.dot,M_list,np.eye(4))

    
class PUMA(SerialLink):
    """
    Parameters from
    
    https://github.com/petercorke/robotics-toolbox-matlab/blob/master/models/mdl_puma560.m
    
    More info at
    https://github.com/petercorke/robotics-toolbox-matlab
    """
    def __init__(self,base=None,tool=None):
        # DH Parameters
        self.d = np.array([0,0,.15,.4318,0,0])
        self.a = np.array([0,.4318,.0203,0,0,0])
        self.alpha = (np.pi/2) * np.array([1,0,-1,1,-1,0])
        
        
        
        Links = [Revolute(d,a,alpha) for d,a,alpha in \
                zip(self.d,self.a,self.alpha)]
        super().__init__(Links,base,tool)
        
        
        # Link Masses
        self.m = np.array([0.,17.4,4.8,0.82,0.34,0.09])
        
        # Link inertias
        # Stored as Ixx,Iyy,Izz,Ixy,Iyz,Ixz
        self.I = np.array([[0,.35,0,0,0,0],
                           [0.13, 0.524, 0.539, 0, 0, 0],
                           [0.066, 0.086, 0.0125, 0, 0, 0],
                           [1.8e-3, 1.3e-3, 1.8e-3, 0, 0, 0],
                           [0.3e-3, 0.4e-3, 0.3e-3, 0, 0, 0],
                           [0.15e-3, 0.15e-3, 0.04e-3, 0, 0, 0]])
        
        # Only works because they are diagonal inertias
        self.I = np.array([np.diag(I[:3]) for I in self.I])
        
        # Location of link center of mass
        self.r = np.array([[0, 0, 0],
                           [-0.3638, 0.006, 0.2275],
                           [-0.0203, -0.0141, 0.070],
                           [0, 0.019, 0],
                           [0, 0, 0],
                           [0, 0, 0.032]])
        
        # Gear ratios
        self.G = np.array([-62.6111,107.815,-53.7063,76.0364,71.923,76.686])
        
        # Actuator inertias
        self.Jm = np.array([200e-6,200e-6,200e-6,33e-6,33e-6,33e-6])
        
        # Viscous Friction of actuator
        self.B = np.array([1.48e-3,.817e-3,1.38e-3,71.2e-6,82.6e-6,36.7e-6])
        
        # Coulomb friction at actuator
        self.Tc = np.array([[0.395, -0.435],
                            [0.126, -0.071],
                            [0.132, -0.105],
                            [11.2e-3, -16.9e-3],
                            [9.26e-3, -14.5e-3],
                            [3.96e-3, -10.5e-3]])
        
        deg = np.pi/180.
        # Joint Limits
        self.qlim = deg * np.array([[-160, 160.],
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
        r = la.norm(p[:2])
        phi = np.arctan2(py,px)
        t_int = np.arcsin(d3/r)
        Theta_1 = [phi+t_int,phi-t_int+np.pi]
        
        # Second angle
        
        Thetas = []
        for theta1 in Theta_1:
            c1 = np.cos(theta1)
            s1 = np.sin(theta1)
            
            V114 = c1 * px + s1 * py
            r = np.sqrt(V114**2 + pz**2)
            
            num = a2**2 - d4**2 - a3**2 + V114**2 + pz**2
            den = 2*a2*r
            psi = np.arccos(num/den)
            phi = np.arctan2(pz,V114)
            
            Theta_2 = [phi+psi,phi-psi]
            for theta2 in Theta_2:
                # theta3
                c2 = np.cos(theta2)
                s2 = np.sin(theta2)
                phi = np.arctan2(a3,d4)
                num = c2 * V114+s2*pz - a2
                den = c2*pz - s2 * V114
                psi = np.arctan2(num,den)
                theta3 = phi-psi
                
                # theta4
                c23 = np.cos(theta2+theta3)
                s23 = np.sin(theta2+theta3)
                V323 = c1 * ay - s1 * ax
                V113 = c1 * ax + s1 * ay
                V313 = c23 * V113 + s23 * az
                
                Theta_4 = [np.arctan2(-V323,-V313),
                           np.arctan2(V323,V313)]
                for theta4 in Theta_4:
                    # theta5
                    c4 = np.cos(theta4)
                    s4 = np.sin(theta4)
                    s5 = -c4 * V313 - s4 * V323
                    c5 = -s23 * V113 + c23 * az
                    theta5 = np.arctan2(s5,c5)
                    
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
                    theta6 = np.arctan2(s6,c6)
                    Thetas.append([theta1,
                                   theta2,
                                   theta3,
                                   theta4,
                                   theta5,
                                   theta6])
                    
        return np.array(Thetas)
    
    def rne(self,q,q_dot,q_ddot,gravity=np.array([0,0,9.81]),
            fext=np.zeros(6)):
        """
        Based on 
        
        Luh, John YS, Michael W. Walker, and Richard PC Paul. 
        "On-line computational scheme for mechanical manipulators." (1980): 69-76.
        
        and the corresponding implementation from
        https://github.com/petercorke/robotics-toolbox-matlab/blob/master/%40SerialLink/rne_dh.m
        
        
        """
        
        n = len(q)
        z0 = np.array([0,0,1.])
        
            
        
        
        # Forward Pass
        Rb = np.eye(3)
        w = np.zeros(3)
        wd = np.zeros(3)
        vd = np.copy(gravity)
           
        Forces = []
        Torques = []
        
        Pstar = []
        Rotations = []
        for i in range(len(q)):
            qi = q[i]
            qdi = q_dot[i]
            qddi = q_ddot[i]
            r = self.r[i]
            
            T = self.Links[i](qi)
            R = T[:3,:3]
            Rotations.append(R)
            Rt = R.T
            
            d = self.d[i]
            alpha = self.alpha[i]
            pstar = np.array([self.a[i],d*np.sin(alpha),d*np.cos(alpha)])
            Pstar.append(pstar)
            
            
            wd = Rt@(wd + z0*qddi + np.cross(w,z0*qdi) )
            w = Rt@(w + z0*qdi)
            vd = np.cross(wd,pstar) + np.cross(w,np.cross(w,pstar)) + Rt@vd
            
            vhat = np.cross(wd,r) + np.cross(w,np.cross(w,r)) + vd
            F = self.m[i] * vhat
            N = self.I[i] @ wd + np.cross(w,self.I[i]@w)
            
            Forces.append(F)
            Torques.append(N)
        # Backward pass
        
        f = fext[:3]
        nn = fext[3:]
        
        jointTorques = []
        for i in range(n)[::-1]:
            pstar = Pstar[i]
            
            if i == n-1:
                R = np.eye(3)
            else:
                R = Rotations[i+1]
                
            
            r = self.r[i]
            
    
            
            nn = R@(nn+np.cross(R.T@pstar,f)) + np.cross(pstar+r,Forces[i]) + Torques[i]
            f = R@f + Forces[i]
            
            
            
            R = Rotations[i]
            
            G = self.G[i]
            Jm = self.Jm[i]
            
            # friction force
            B = self.B[i]
            Tc = self.Tc[i]
            tau = B * np.abs(G) * q_dot[i]
            
            # This is coulomb friction
            # It causes lots of problems in simulation
            #if q_dot[i] > 0:
            #    tau = tau + Tc[0]
            #elif q_dot[i] < 0:
            #    tau = tau + Tc[1]
                
            tau = -np.abs(G) * tau

            # Total torque
            t = nn@R.T@z0 + G**2 * Jm * q_ddot[i] - tau
            jointTorques.append(t)
        return np.array(jointTorques[::-1])
    def MassMatrix(self,q):
        MM = []
        for qdd in np.eye(len(q)):
            tau = self.rne(q,np.zeros(len(q)),qdd,gravity=np.zeros(3))
            MM.append(tau)
            
        return np.array(MM)
    
    def accel(self,q,q_dot,tau):
        M = self.MassMatrix(q)
        tau_grav = self.rne(q,q_dot,np.zeros_like(q))
        
        q_ddot = la.solve(M,tau-tau_grav)
        return q_ddot
        
    def vectorField(self,x,tau):
        q = x[:6]
        q_dot = x[6:]
        q_ddot = self.accel(q,q_dot,tau)
        return np.hstack([q_dot,q_ddot])

    def grav_torque(self,q):
        return self.rne(q,np.zeros_like(q),np.zeros_like(q))
