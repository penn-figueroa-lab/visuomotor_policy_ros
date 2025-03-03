import numpy as np
import scipy.spatial.transform as scipy_transform

"""
This is a library of spatial math utilities.
All interfaces are numpy array based, and the last dimension(s) is always the vector dimension.

p: 3D vector representing position
SE3: 4x4 matrix representing rigid body transformation
SO3, or R: 3x3 matrix representing rotation
quat: 4D vector representing quaternion, (w, x, y, z)
twist: 6D vector representing a twist
pose7: 7D vector representing a pose, (x, y, z, qw, qx, qy, qz)
pose9: 9D vector representing a pose, (x, y, z, r1, r2, r3, r4, r5, r6)
rot6: 6D vector representing a rotation, (r1, r2, r3, r4, r5, r6)
"""

# Basic operations


def transpose(mat):
    """
    Transpose the last two dimensions of a batch of matrices.

    :param      mat:  (..., h, w) np array
    :return:     (...,  w, h) np array, transposed matrices
    """
    return np.swapaxes(mat, -1, -2)


def normalize(vec, eps=1e-12):
    """
    Normalize vectors along the last dimension.
    args:
        vec: (..., N) np array
    return:
        (..., N) np array of the same shape
    """
    norm = np.linalg.norm(vec, axis=-1)  # (...)
    norm = np.maximum(norm, eps)
    out = vec / norm[..., np.newaxis]
    return out


# type specific operations


def wedge3(vec):
    """
    Compute the skew-symmetric wedge matrix of a batch of 3D vectors.

    :param      vec:  (..., 3) np array
    :return:     (..., 3, 3) np array, skew-symmetric matrix
    """
    shape = vec.shape[:-1]
    out = np.zeros(shape + (3, 3), dtype=vec.dtype)
    out[..., 0, 1] = -vec[..., 2]
    out[..., 0, 2] = vec[..., 1]
    out[..., 1, 2] = -vec[..., 0]

    out[..., 1, 0] = -out[..., 0, 1]
    out[..., 2, 0] = -out[..., 0, 2]
    out[..., 2, 1] = -out[..., 1, 2]
    return out


def wedge6(vec):
    """
    Compute the homogeneous coordinates of a batch of twists.

    :param      vec:  (..., 6) np array, (v, w)
    :return:     (..., 4, 4) np array
    """
    shape = vec.shape[:-1]
    out = np.zeros(shape + (4, 4), dtype=vec.dtype)
    out[..., :3, :3] = wedge3(vec[..., 3:])
    out[..., :3, 3] = vec[..., :3]
    return out


def SE3_inv(mat):
    """
    Efficient inverse of a batch of SE3 matrices.

    Tested by generating random SE3 and verify SE3_inv(SE3) @ SE3 = Identity.

    :param      mat:  (..., 4, 4) np array
    :return:     (..., 4, 4) np array, inverse of the input matrix
    """
    SE3_inv = np.zeros_like(mat)
    SE3_inv[..., :3, :3] = transpose(mat[..., :3, :3])

    temp = -SE3_inv[..., :3, :3] @ np.expand_dims(mat[..., :3, 3], -1)
    SE3_inv[..., :3, 3] = temp.squeeze()
    SE3_inv[..., 3, 3] = 1
    return SE3_inv


# transformations
def trans_p_by_SE3(p, SE3):
    """
    Transform a batch of 3D points by a batch of SE3 matrices.

    :param      p:    (..., 3) np array, 3D points
    :param      SE3:  (..., 4, 4) np array, SE3 matrices
    :return:     (..., 3) np array, transformed points
    """
    p = np.expand_dims(p, -2)
    return np.sum(SE3[..., :3, :3] * p, axis=-1) + SE3[..., :3, 3]


# Type conversions


def JacTwist2BodyV(R):
    """
    From a SO3, compute the Jacobian matrix that maps twist to body velocity.

    :param      R:  (3, 3) np array, the rotation matrix
    :return:     (6, 6) np array, the Jacobian matrix
    """

    Jac = np.eye(6)
    Jac[3, 3] = R[0, 2] * R[0, 2] + R[1, 2] * R[1, 2] + R[2, 2] * R[2, 2]
    Jac[3, 5] = -R[0, 0] * R[0, 2] - R[1, 0] * R[1, 2] - R[2, 0] * R[2, 2]
    Jac[4, 3] = -R[0, 0] * R[0, 1] - R[1, 0] * R[1, 1] - R[2, 0] * R[2, 1]
    Jac[4, 4] = R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0] + R[2, 0] * R[2, 0]
    Jac[5, 4] = -R[0, 1] * R[0, 2] - R[1, 1] * R[1, 2] - R[2, 1] * R[2, 2]
    Jac[5, 5] = R[0, 1] * R[0, 1] + R[1, 1] * R[1, 1] + R[2, 1] * R[2, 1]

    return Jac


def pose7_to_SE3(pose7):
    """
    pose7: [:, 7] with x,y,z,qw,qx,qy,qz
    returns: [:, 4, 4] SE3 matrices
    """
    qw = pose7[..., 3]
    qx = pose7[..., 4]
    qy = pose7[..., 5]
    qz = pose7[..., 6]
    q11 = qx * qx
    q22 = qy * qy
    q33 = qz * qz
    q01 = qw * qx
    q02 = qw * qy
    q03 = qw * qz
    q12 = qx * qy
    q13 = qx * qz
    q23 = qy * qz

    shape = pose7.shape[:-1]
    SE3 = np.zeros(shape + (4, 4), dtype=pose7.dtype)
    SE3[..., 0, 0] = 1.0 - 2.0 * q22 - 2.0 * q33
    SE3[..., 0, 1] = 2.0 * (q12 - q03)
    SE3[..., 0, 2] = 2.0 * (q13 + q02)
    SE3[..., 1, 0] = 2.0 * (q12 + q03)
    SE3[..., 1, 1] = 1.0 - 2.0 * q11 - 2.0 * q33
    SE3[..., 1, 2] = 2.0 * (q23 - q01)
    SE3[..., 2, 0] = 2.0 * (q13 - q02)
    SE3[..., 2, 1] = 2.0 * (q23 + q01)
    SE3[..., 2, 2] = 1.0 - 2.0 * q11 - 2.0 * q22

    SE3[..., :3, 3] = pose7[..., :3]

    SE3[..., 3, 3] = 1

    return SE3


def pose9_to_SE3(d9):
    p = d9[..., :3]
    d6 = d9[..., 3:]
    R = rot6_to_SO3(d6)
    SE3 = np.zeros(d9.shape[:-1] + (4, 4), dtype=d9.dtype)
    SE3[..., :3, :3] = R
    SE3[..., :3, 3] = p
    SE3[..., 3, 3] = 1
    return SE3


def quat_to_aa(quat, tol=1e-7):
    """
    (not vectorized)
    Convert a quaternion to axis-angle representation.

    :param      quat:  (4,) np array
    :return:     (4,) np array, axis-angle representation (axis, angle)
    """
    angle = 2 * np.arccos(quat[..., 0])

    axis = quat[1:]
    axis_norm = np.linalg.norm(axis)

    if axis_norm < tol:
        return np.array([1, 0, 0, 0], dtype=quat.dtype)
    axis /= axis_norm

    return np.array([*axis, angle], dtype=quat.dtype)


def rot6_to_SO3(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    SO3 = np.stack((b1, b2, b3), axis=-2)
    return SO3


def SE3_to_adj(SE3):
    """
    Compute the adjoint matrix of a batch of SE3 matrices.

    :param      SE3:  (..., 4, 4) np array
    :return:     (..., 6, 6) np array, adjoint matrices
    """
    shape = SE3.shape[:-2]
    R = SE3[..., :3, :3]
    p = SE3[..., :3, 3]
    Adj = np.zeros(shape + (6, 6), dtype=SE3.dtype)
    Adj[..., :3, :3] = R
    Adj[..., :3, 3:] = wedge3(p) @ R
    Adj[..., 3:, 3:] = R
    return Adj


def SE3_to_pose7(SE3):
    p = SE3[..., :3, 3]
    R = SE3[..., :3, :3]
    q = SO3_to_quat(R)
    pose7 = np.concatenate([p, q], axis=-1)
    return pose7


def SE3_to_pose9(SE3):
    p = SE3[..., :3, 3]
    R = SE3[..., :3, :3]
    d6 = SO3_to_rot6d(R)
    pose9 = np.concatenate([p, d6], axis=-1)
    return pose9


def SE3_to_se3(SE3, kEpsilon=1e-7):
    """displacement to twist coordinate (se3)
    :param      SE3:  (4, 4) np array
    :return      (6) np array, twist coordinates
    """
    assert SE3.shape == (4, 4)
    p = SE3[:3, 3]
    R = SE3[:3, :3]
    omega = SO3_to_so3(R, kEpsilon)
    theta = np.linalg.norm(omega, axis=-1, keepdims=True)
    if theta < kEpsilon:
        return np.concatenate([p, omega], axis=-1)
    omega /= theta
    M = (np.eye(3) - R) @ wedge3(omega) + omega @ omega.T * theta
    se3 = np.zeros(SE3.shape[:-2] + (6,), dtype=SE3.dtype)
    se3[:3] = np.linalg.solve(M, p)
    se3[3:] = omega
    se3 *= theta
    return se3


def SE3_to_spt(SE3):
    """displacement to special twist coordinate
    :param      SE3:  (..., 4, 4) np array
    :return      (..., 6) np array, twist coordinates
    """
    twist_coordinate = np.zeros(SE3.shape[:-2] + (6,), dtype=SE3.dtype)
    twist_coordinate[..., :3] = SE3[..., :3, 3]
    twist_coordinate[..., 3:] = SO3_to_so3(SE3[..., :3, :3])
    return twist_coordinate


def se3_to_SE3(se3, kEpsilon=1e-9):
    """twist coordinate to displacement
    :param      se3:  (6,) np array, twist coordinates
    :return      (4, 4) np array
    """
    if se3.shape == (6, 1):
        se3 = se3.reshape(6)  # (6,1) -> (6,)
    if se3.shape != (6,):
        raise ValueError(f"se3 shape should be (6, 1) or (6,), got {se3.shape}")

    v = se3[:3]
    w = se3[3:]
    theta = np.linalg.norm(w)

    if np.fabs(theta) < kEpsilon:
        SE3 = np.eye(4)
        SE3[:3, 3] = v
    else:
        v /= theta
        w /= theta
        R = so3_to_SO3(w)
        SE3 = np.eye(4)
        SE3[:3, :3] = R
        SE3[:3, 3] = (np.eye(3) - R) @ np.cross(w, v) + w * w.T @ v * theta
    return SE3


def SO3_to_rot6d(mat):
    batch_dim = mat.shape[:-2]
    out = mat[..., :2, :].copy().reshape(batch_dim + (6,))
    return out


def SO3_to_so3(R, kEpsilon=1e-7):
    """Get exponential coordinate of a rotation matrix

    :param      R:    (3, 3) numpy array

    :returns:   (3,) numpy array
    """
    assert R.shape == (3, 3)
    dim = len(R.shape)
    output_shape = R.shape[:-2] + (3,)
    temp_arg_to_cos = (np.trace(R, axis1=dim - 2, axis2=dim - 1) - 1.0) / 2.0
    # truncate temp_arg_to_cos between  -1.0, 1.0
    temp_arg_to_cos = np.maximum(np.minimum(temp_arg_to_cos, 1), -1)
    theta = np.arccos(temp_arg_to_cos)
    if np.fabs(theta) < kEpsilon:
        so3 = np.broadcast_to([1.0, 0.0, 0.0], output_shape).copy()
    else:
        so3 = np.array(
            [
                R[..., 2, 1] - R[..., 1, 2],
                R[..., 0, 2] - R[..., 2, 0],
                R[..., 1, 0] - R[..., 0, 1],
            ]
        )
        so3 /= 2.0 * np.sin(theta)
    so3 *= theta
    return so3


def so3_to_SO3(v, kEpsilon=1e-9):
    """Get rotation matrix from exponential coordinate

    :param      v:    (..., 3) numpy array

    :returns:   (..., 3, 3) numpy array

    Needs to be tested
    """
    theta = np.linalg.norm(v, axis=-1, keepdims=True)
    theta = np.maximum(theta, kEpsilon)
    vn = v / theta
    v_wedge = wedge3(vn)
    SO3 = (
        np.eye(3) + v_wedge * np.sin(theta) + v_wedge @ v_wedge * (1.0 - np.cos(theta))
    )
    return SO3


def SO3_to_quat(R):
    """Convert rotation matrix to quaternion

    :param      R:  (..., 3, 3) np array
    :return:     (..., 4) np array, quaternion
    """
    scipy_quat = scipy_transform.Rotation.from_matrix(R).as_quat()
    return scipy_quat[..., [3, 0, 1, 2]]


def spt_to_SE3(twist):
    """special twist coordinate to displacement
    :param      twist:  (..., 6) np array, twist coordinates
    :return      (..., 4, 4) np array
    """
    if twist.shape == (6, 1):
        twist = twist.reshape(6)  # (6,1) -> (6,)
    if twist.shape[-1] != 6:
        raise ValueError(f"twist shape should be (..., 6) or (6,), got {twist.shape}")
    SE3 = np.zeros(twist.shape[:-1] + (4, 4), dtype=twist.dtype)
    SE3[..., :3, :3] = so3_to_SO3(twist[..., 3:])
    SE3[..., :3, 3] = twist[..., :3]
    SE3[..., 3, 3] = 1
    return SE3


def twc_to_SE3(twc):
    """twist coordinate to displacement
    :param      twc:  (..., 6) np array, twist coordinates
    :return      (..., 4, 4) np array

    Needs to be tested
    """
    if twc.shape == (6, 1):
        twc = twc.reshape(6)  # (6,1) -> (6,)
    if twc.shape[-1] != 6:
        raise ValueError(f"twc shape should be (..., 6) or (6,), got {twc.shape}")
    SE3 = np.zeros(twc.shape[:-1] + (4, 4), dtype=twc.dtype)
    SE3[..., :3, :3] = so3_to_SO3(twc[..., 3:])
    SE3[..., :3, 3] = twc[..., :3]
    SE3[..., 3, 3] = 1
    return SE3


## Legacy code from UMI

# def pos_rot_to_mat(pos, rot):
#     shape = pos.shape[:-1]
#     mat = np.zeros(shape + (4,4), dtype=pos.dtype)
#     mat[...,:3,3] = pos
#     mat[...,:3,:3] = rot.as_matrix()
#     mat[...,3,3] = 1
#     return mat

# def mat_to_pos_rot(mat):
#     pos = (mat[...,:3,3].T / mat[...,3,3].T).T
#     rot = st.Rotation.from_matrix(mat[...,:3,:3])
#     return pos, rot

# def pos_rot_to_pose(pos, rot):
#     shape = pos.shape[:-1]
#     pose = np.zeros(shape+(6,), dtype=pos.dtype)
#     pose[...,:3] = pos
#     pose[...,3:] = rot.as_rotvec()
#     return pose

# def pose_to_pos_rot(pose):
#     pos = pose[...,:3]
#     rot = st.Rotation.from_rotvec(pose[...,3:])
#     return pos, rot

# def pose_to_mat(pose):
#     return pos_rot_to_mat(*pose_to_pos_rot(pose))
