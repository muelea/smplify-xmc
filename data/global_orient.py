import numpy as np

# map global orientation code to angle (in radians)
# global orientations to select the inital global orientation from

go_map = {
    'frontal': 3.1415927 * np.array([1, 0, 0]),
    '0': 3.1415927 * np.array([-0.7071068, 0, 0.7071068]),
    '1': 3.1415927 * np.array([ -0.8660254, 0, 0.5 ]),
    '2': 3.1415927 * np.array([ -0.9659258, 0, 0.258819 ]),
    '3': 3.1415927 * np.array([1, 0, 0]), # frontal
    '4': 3.1415927 * np.array([ 0.9659258, 0, 0.258819 ]),
    '5': 3.1415927 * np.array([ 0.8660254, 0, 0.5 ]),
    '6': 3.1415927 * np.array([0.7071068, 0, 0.7071068]),
    '7':  3.1415927 * np.array([0.5, 0, 0.8660254]),
    '8':  3.1415927 * np.array([0.258819, 0, 0.9659258]),
    '9':  3.1415927 * np.array([0,0,1]),
    '10': 3.1415927 * np.array([-0.5, 0, 0.8660254]),
    '11': 3.1415927 * np.array([-0.7071068, 0, 0.7071068])
}
