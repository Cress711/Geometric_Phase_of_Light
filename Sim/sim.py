import numpy as np
import matplotlib.pyplot as plt

from optical_elements import QWP, HWP, qwp_theta_to_return_to_RCP


def RCP():
    return np.array([0.0, 0.0, 1.0])


def LCP():
    return np.array([0.0, 0.0, -1.0])


def simulate_QWP_only(steps=300):
    S = RCP()
    traj = [S]

    t, S = QWP(S, theta=0.0, steps=steps)
    traj.extend(t)

    return np.array(traj)


def simulate_HWP_only(theta_hwp, steps=300):
    S = RCP()
    traj = [S]

    t, S = HWP(S, theta=theta_hwp, steps=steps, path="physical")
    traj.extend(t)

    return np.array(traj)


def simulate_QWP_HWP_QWP(theta_hwp, steps=300):
    S = RCP()
    traj = [S]

    # 1. QWP: biegun -> równik
    t, S = QWP(S, theta=0.0, steps=steps)
    traj.extend(t)

    # 2. HWP: ruch po równiku
    t, S = HWP(S, theta=theta_hwp, steps=steps, path="equator")
    traj.extend(t)

    # 3. QWP: równik -> RCP
    theta_qwp2 = qwp_theta_to_return_to_RCP(S)

    t, S = QWP(S, theta=theta_qwp2, steps=steps)
    traj.extend(t)

    return np.array(traj)


def plot_poincare():
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    ax.plot_surface(x, y, z, alpha=0.12, color="gray")

    ax.set_xlabel("S1")
    ax.set_ylabel("S2")
    ax.set_zlabel("S3")

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect([1, 1, 1])

    return fig, ax


steps = 300

angles = [
    0,
]

fig, ax = plot_poincare()

for theta in angles:
    traj = simulate_QWP_HWP_QWP(theta_hwp=theta, steps=steps)

    ax.plot(
        traj[:, 0],
        traj[:, 1],
        traj[:, 2],
        label=f"γ = {theta:.2f} rad"
    )

ax.scatter(*RCP(), color="blue", s=60, label="RCP")
ax.scatter(*LCP(), color="red", s=60, label="LCP")

ax.legend()
plt.title("Trajektorie QWP-HWP-QWP na sferze Poincare")
plt.show()