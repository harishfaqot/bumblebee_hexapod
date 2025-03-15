import pybullet as p
import pybullet_data
import numpy as np

def add_terrain():
    # Heightfield parameters (smaller terrain)
    size = 10  # 64x64 grid
    height_scale = np.random.uniform(0, 0.15)  # Controls terrain roughness

    # Generate small random terrain
    terrain_shape = np.random.uniform(low=-1, high=1, size=(size, size))
    terrain_shape = (terrain_shape / np.max(np.abs(terrain_shape))).flatten()

    # Create heightfield
    terrain_id = p.createCollisionShape(
        shapeType=p.GEOM_HEIGHTFIELD,
        meshScale=[0.5, 0.5, height_scale],  # Smaller scale
        heightfieldTextureScaling=(size - 1) / 2,
        heightfieldData=terrain_shape,
        numHeightfieldRows=size,
        numHeightfieldColumns=size
    )

    # Create terrain body
    terrain_body = p.createMultiBody(0, terrain_id)
    p.changeVisualShape(terrain_body, -1, rgbaColor=[0.3, 0.3, 0.7, 1])  # Greenish terrain

    return terrain_body

if __name__ == '__main__':
    # Initialize physics engine
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")

    # Load terrain
    terrain = add_terrain()

    # Run simulation loop
    while True:
        p.stepSimulation()
