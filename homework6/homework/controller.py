import pystk
import numpy as np

def control(aim_point, current_vel):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in local coordinate frame
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    action = pystk.Action()

    """
    Your code here
    Hint: Use action.acceleration (0..1) to change the velocity. Try targeting a target_velocity (e.g. 20).
    Hint: Use action.brake to True/False to brake (optionally)
    Hint: Use action.steer to turn the kart towards the aim_point, clip the steer angle to -1..1
    Hint: You may want to use action.drift=True for wide turns (it will turn faster)
    """
    
    terminal_vel = 20
    # print(aim_point)
    steer_angle = np.clip(aim_point[0], -1, 1) #np.arcsin(aim_point[0] / aim_point[2]) * aim_point[0]
    if np.abs(aim_point[0]) > 7 and current_vel > 10 and np.sign(action.steer) == np.sign(steer_angle):
        action.acceleration = -1
        action.brake = True
        action.steer = action.steer + steer_angle
        action.drift = True
    elif np.abs(aim_point[0]) > 5 and current_vel > 10:
        action.brake = True
        action.acceleration = -1
        action.steer = action.steer + steer_angle
    elif np.abs(aim_point[0]) > 5:
        action.brake = False
        if current_vel < 7:
            action.acceleration = 1
        action.steer = action.steer + steer_angle
    elif current_vel < terminal_vel:
        action.brake = False
        action.acceleration = 1
    elif current_vel > terminal_vel:
        action.acceleration = -1
    action.steer = action.steer + steer_angle
        

    return action


if __name__ == '__main__':
    from .utils import PyTux
    from argparse import ArgumentParser


    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(steps)
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
