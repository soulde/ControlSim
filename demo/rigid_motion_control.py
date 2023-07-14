import math
import numpy as np
import quaternion
from control_env.implement.RigidBody import RigidBody
import matplotlib.pyplot as plt


def main():
    init = quaternion.from_euler_angles(np.array([1., 21., 50]) / 180 * np.pi)
    rb = RigidBody(np.array([init.w, init.x, init.y, init.z]))
    print(init)
    target = quaternion.from_euler_angles(np.array([30., 30., 1.]) / 180 * np.pi)
    print(target)
    u = np.array([0, 0, 0])
    for i in range(500):
        q = rb(u, 0.01)
        norm_q = quaternion.from_float_array(q).normalized()
        delta_q = norm_q.inverse() * target
        angle = math.acos(delta_q.w) * 2
        if angle > np.pi / 2:
            angle = np.pi - angle
            delta_q.w = math.cos(angle / 2)
            delta_q.x = -delta_q.x
            delta_q.y = -delta_q.y
            delta_q.z = -delta_q.z
        u = np.array([delta_q.x, delta_q.y, delta_q.z]) * (1 - delta_q.w) * 100
    t_array, x_array = rb.history()

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    s = np.linspace(0, 1, 2)

    for v in map(lambda k: quaternion.as_rotation_vector(quaternion.from_float_array(k)), x_array):
        x, y, z = quaternion.as_euler_angles(quaternion.from_rotation_vector(v))

        X = s * x
        Y = s * y
        Z = s * z

        ax.plot3D(X, Y, Z, 'blue')
        plt.pause(.001)


if __name__ == '__main__':
    main()
