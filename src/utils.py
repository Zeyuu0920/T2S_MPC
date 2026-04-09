import numpy as np
import pybullet as pb


def get_pb_handles(env):
    """
    Return: (cid, robot_id)
      cid: physicsClientId (int) bound to env's pybullet client
      robot_id: drone body unique id (int)
    """
    cid = None
    cid_candidates = [
        "PYB_CLIENT", "pyb_client", "_pyb_client", "client", "CLIENT",
        "physics_client_id", "_physics_client_id", "physicsClientId"
    ]
    for name in cid_candidates:
        if hasattr(env, name):
            v = getattr(env, name)
            if isinstance(v, int):
                cid = v
                break
            if hasattr(v, "_client") and isinstance(v._client, int):
                cid = v._client
                break
            if hasattr(v, "id") and isinstance(v.id, int):
                cid = v.id
                break

    if cid is None:
        cid = 0

    robot_id = None
    rid_candidates = ["DRONE_IDS", "drone_ids", "DRONE_ID", "drone_id", "ROBOT_ID", "robot_id"]
    for name in rid_candidates:
        if hasattr(env, name):
            v = getattr(env, name)
            if isinstance(v, (list, tuple)) and len(v) > 0:
                robot_id = int(v[0])
                break
            if isinstance(v, int):
                robot_id = int(v)
                break

    if robot_id is None:
        nb = pb.getNumBodies(physicsClientId=cid)
        if nb <= 0:
            raise RuntimeError(
                f"[pybullet] No bodies in client {cid}. "
                "This usually means you're using the wrong physicsClientId."
            )
        robot_id = pb.getBodyUniqueId(nb - 1, physicsClientId=cid)

    return cid, robot_id


def get_mass(cid, robot_id):
    return pb.getDynamicsInfo(robot_id, -1, physicsClientId=cid)[0]


