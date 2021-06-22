# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2021 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de


defaultweights = {

        'data_weights': [1.0, 1.0, 1.0],
        'body_pose_prior_weights': [10.0, 5.0],
        'mimicked_contact_weights': [30000, 30000],
        'scopti_weights': [0.0, 500.0],
        'coll_loss_weights': [0.0, 1.0],
        'shape_weights': [0.5e1, 0.5e1],
        'expr_weights': [0.5e1, 0.5e1],
        'hand_pose_prior_weights': [4.0, 4.0],
        'jaw_pose_prior_weights':[[47.8,478,478], [47.8,478,478]],
        'hand_joints_weights': [2.0, 2.0],
        'face_joints_weights': [2.0, 2.0],

}

def load_weights(data_weights,
                 body_pose_prior_weights,
                 use_hands,
                 hand_pose_prior_weights,
                 hand_joints_weights,
                 shape_weights,
                 use_face,
                 jaw_pose_prior_weights,
                 expr_weights,
                 face_joints_weights,
                 scopti_weights,
                 mimicked_contact_weights,
                 camera_rot_weights,
                 body_global_orient_weights,
                 focal_length_weights,
                 use_contact,
                 **kwargs):

    # get correct default weights
    weights = defaultweights

    # set weights
    if data_weights is None:
        data_weights = weights['data_weights']

    if body_pose_prior_weights is None:
        body_pose_prior_weights = weights['body_pose_prior_weights']
    msg = (
        'Number of Body pose prior weights {}'.format(
            len(body_pose_prior_weights)) +
        ' does not match the number of data term weights {}'.format(
            len(data_weights)))
    assert (len(data_weights) ==
            len(body_pose_prior_weights)), msg

    if use_hands:
        if hand_pose_prior_weights is None:
            hand_pose_prior_weights = weights['hand_pose_prior_weights']
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of hand pose prior weights')
        assert (len(hand_pose_prior_weights) ==
                len(body_pose_prior_weights)), msg
        if hand_joints_weights is None:
            hand_joints_weights = weights['hand_joints_weights']
            msg = ('Number of Body pose prior weights does not match the' +
                   ' number of hand joint distance weights')
            assert (len(hand_joints_weights) ==
                    len(body_pose_prior_weights)), msg

    if shape_weights is None:
        shape_weights = weights['shape_weights']
    msg = ('Number of Body pose prior weights = {} does not match the' +
           ' number of Shape prior weights = {}')
    assert (len(shape_weights) ==
            len(body_pose_prior_weights)), msg.format(
                len(shape_weights),
                len(body_pose_prior_weights))

    if use_face:
        if jaw_pose_prior_weights is None:
            jaw_pose_prior_weights = weights['jaw_pose_prior_weights']
        else:
            jaw_pose_prior_weights = map(lambda x: map(float, x.split(',')),
                                         jaw_pose_prior_weights)
            jaw_pose_prior_weights = [list(w) for w in jaw_pose_prior_weights]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of jaw pose prior weights')
        assert (len(jaw_pose_prior_weights) ==
                len(body_pose_prior_weights)), msg

        if expr_weights is None:
            expr_weights = weights['expr_weights']
        msg = ('Number of Body pose prior weights = {} does not match the' +
               ' number of Expression prior weights = {}')
        assert (len(expr_weights) ==
                len(body_pose_prior_weights)), msg.format(
                    len(body_pose_prior_weights),
                    len(expr_weights))

        if face_joints_weights is None:
            face_joints_weights = weights['face_joints_weights']
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of face joint distance weights')
        assert (len(face_joints_weights) ==
                len(body_pose_prior_weights)), msg

    if use_contact:
        if scopti_weights is None:
            scopti_weights = weights['scopti_weights']
        msg = ('Number of Body selfcontact optimization weights does not match the' +
            ' number of collision loss weights')
        assert (len(scopti_weights) ==
                len(body_pose_prior_weights)), msg

        if mimicked_contact_weights is None:
            mimicked_contact_weights = weights['mimicked_contact_weights']
        msg = ('Number of Body selfcontact optimization weights does not match the' +
            ' number of collision loss weights')
        assert (len(mimicked_contact_weights) ==
                len(body_pose_prior_weights)), msg

    return create_weights_dict(
        data_weights, body_pose_prior_weights, hand_pose_prior_weights,
        hand_joints_weights, shape_weights, jaw_pose_prior_weights,
        expr_weights, face_joints_weights, scopti_weights, mimicked_contact_weights,
        camera_rot_weights, body_global_orient_weights, focal_length_weights,
        use_face, use_hands, use_contact,
    )


def create_weights_dict(
    data_weights, body_pose_prior_weights, hand_pose_prior_weights,
    hand_joints_weights, shape_weights, jaw_pose_prior_weights,
    expr_weights, face_joints_weights, scopti_weights, mimicked_contact_weights,
    camera_rot_weights, body_global_orient_weights, focal_length_weights,
    use_face, use_hands, use_contact
):
   # Weights used for the pose prior and the shape prior
    opt_weights_dict = {'data_weight': data_weights,
                        'body_pose_weight': body_pose_prior_weights,
                        'shape_weight': shape_weights,
                        'camera_rot_weight': camera_rot_weights,
                        'body_global_orient_weight': body_global_orient_weights,
                        'focal_length_weight': focal_length_weights}
    if use_face:
        opt_weights_dict['face_weight'] = face_joints_weights
        opt_weights_dict['expr_prior_weight'] = expr_weights
        opt_weights_dict['jaw_prior_weight'] = jaw_pose_prior_weights
    if use_hands:
        opt_weights_dict['hand_weight'] = hand_joints_weights
        opt_weights_dict['hand_prior_weight'] = hand_pose_prior_weights
    if use_contact:
        opt_weights_dict['mimicked_contact_weight'] = mimicked_contact_weights
        opt_weights_dict['scopti_weight'] = scopti_weights

    keys = opt_weights_dict.keys()

    opt_weights = [dict(zip(keys, vals)) for vals in
                   zip(*(opt_weights_dict[k] for k in keys
                         if opt_weights_dict[k] is not None))]

    return opt_weights