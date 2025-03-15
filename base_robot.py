import pybullet as p
import pybullet_data
import time

# Start PyBullet in GUI mode
physicsClient = p.connect(p.GUI)

# Set search path for assets
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load the URDF model with STL/OBJ reference
stl_id = p.loadURDF("models/krsri2024.urdf", basePosition=[-3, 0, 0], useFixedBase=True)
print(f"Loaded model ID: {stl_id}")

# Set gravity (optional, adjust as needed)
p.setGravity(0, 0, -9.8)

# Run simulation loop
while True:
    p.stepSimulation()
    time.sleep(1./240.)  # PyBullet runs at 240Hz
