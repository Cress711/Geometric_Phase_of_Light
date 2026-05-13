import numpy as np
import matplotlib.pyplot as plt

def R(theta):
    """Macierz rotacji"""
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

def HWP(theta):
    """Macierz półfalówki obróconej o kąt theta"""
    return R(-theta) @ np.array([[1, 0], [0, -1]]) @ R(theta)

def jones_RCP():
    return (1/np.sqrt(2)) * np.array([1, -1j])

def jones_LCP():
    return (1/np.sqrt(2)) * np.array([1, 1j])

def stokes_vector(E):
    Ex, Ey = E[0], E[1]
    S0 = np.abs(Ex)**2 + np.abs(Ey)**2
    S1 = np.abs(Ex)**2 - np.abs(Ey)**2
    S2 = 2*np.real(np.conj(Ex)*Ey)
    S3 = 2*np.imag(np.conj(Ex)*Ey)
    return np.array([S1, S2, S3]) / S0

def trajectory(theta, steps=200):
    """Symulujemy 'płynne przejście' jako rotację fazową 0→π"""
    psi_in = jones_RCP()
    
    traj = []
    
    for delta in np.linspace(0, np.pi, steps):
        J = R(-theta) @ np.array([[1, 0], [0, np.exp(1j*delta)]]) @ R(theta)
        psi_out = J @ psi_in
        S = stokes_vector(psi_out)
        traj.append(S)
    
    return np.array(traj)

def plot_poincare():
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')

    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    ax.plot_surface(x, y, z, alpha=0.1, color='gray')

    ax.set_xlabel('S1')
    ax.set_ylabel('S2')
    ax.set_zlabel('S3')

    return fig, ax

angles = [0, np.pi/8, np.pi/4, 3*np.pi/8]

fig, ax = plot_poincare()

for theta in angles:
    traj = trajectory(theta)
    ax.plot(traj[:,0], traj[:,1], traj[:,2], label=f'γ={theta:.2f}')

S_R = stokes_vector(jones_RCP())
S_L = stokes_vector(jones_LCP())

ax.scatter(*S_R, color='blue', s=50, label='RCP')
ax.scatter(*S_L, color='red', s=50, label='LCP')

ax.legend()
plt.title("Trajektorie na sferze Poincare (RCP → LCP przez HWP)")
plt.show()
