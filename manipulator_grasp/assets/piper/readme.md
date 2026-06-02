# Piper MuJoCo model

This folder contains a simplified Piper 6-DOF MJCF model used by the local
grasping demo.

It keeps the same important names as the Piper URDF:

- arm joints: `joint1` ... `joint6`
- gripper joints: `joint7`, `joint8`
- base body: `base_link`
- arm tip body: `link6`
- gripper TCP body/site: `gripper_center_link` / `tcp`

If you want visual fidelity, convert your URDF + meshes to MJCF and replace
`piper.xml`, but keep these names or update `piper_grasp_env.py` accordingly.

