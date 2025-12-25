import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from mpl_toolkits.mplot3d import Axes3D

from dataset import FULL_BODY_KEYS, KINEMATIC_TREE


def draw_3d_pose(ax, keypoints3d, color='green', linecolor='black'):
    
    # draw keypoints
    for x, y, z in keypoints3d:
        ax.scatter(x, z, y, c=color, marker='o')

    # draw connections
    for chain in KINEMATIC_TREE:
        for j in range(len(chain)-1):
            start_idx, end_idx = FULL_BODY_KEYS.index(chain[j]), FULL_BODY_KEYS.index(chain[j+1])
            
            if start_idx is not None and end_idx is not None:
                start_point = keypoints3d[start_idx]
                end_point = keypoints3d[end_idx]
                ax.plot(
                    [start_point[0], end_point[0]],  # x
                    [start_point[2], end_point[2]],  # z
                    [start_point[1], end_point[1]],  # y
                    c=linecolor  
                )
    return ax


def save_animation(name, x_iobt, x_tdpt, y_pred, y_true, delay_sec=0.1, frame_rate=30):

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(221, projection='3d')  # TDPT
    ax2 = fig.add_subplot(222, projection='3d')  # IOBT
    ax3 = fig.add_subplot(223, projection='3d')  # Ground Truth
    ax4 = fig.add_subplot(224, projection='3d')  # Prediction

    def animate(i):
        scale=1.6
        
        ax1.cla()
        ax2.cla()
        ax3.cla()
        ax4.cla()

        # axes setting
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xlim(scale/2, -scale/2)
            ax.set_ylim(-scale/2, scale/2)
            ax.set_zlim(0, scale)
            ax.set_xticks(np.linspace(-scale/2, scale/2, 5))
            ax.set_yticks(np.linspace(-scale/2, scale/2, 5))
            ax.set_zticks(np.linspace(0, scale, 5))

        # plot x_tdpt
        draw_3d_pose(ax1, x_tdpt[i], color='blue', linecolor='blue')
        ax1.set_title("TDPT")

        # plot x_iobt
        draw_3d_pose(ax2, x_iobt[i], color='green', linecolor='green')
        ax2.set_title("IOBT (Upper Body)")
        
        # plot y_true
        draw_3d_pose(ax3, y_true[i], color='black', linecolor='black')
        ax3.set_title("Ground Truth")

        # plot y_pred
        draw_3d_pose(ax4, y_pred[i], color='red', linecolor='red')
        ax4.set_title(f"Prediction with {int(delay_sec*1000)}ms Delay")

        return ax

    anim = animation.FuncAnimation(fig, animate, frames=range(0, len(y_true), 1), interval=int(1000/frame_rate), blit=False)
    
    # save animation
    os.makedirs("animations", exist_ok=True)
    anim.save(f'animations/{name}_{int(delay_sec*1000)}ms_dlsa_v4.gif', writer='pillow', fps=frame_rate)