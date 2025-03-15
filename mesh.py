import trimesh

mesh = trimesh.load_mesh("models/krsri2024_fus.stl")

# Print the current face count
print(f"Original face count: {len(mesh.faces)}")

# Set a reasonable target count (e.g., half the original)
target_faces = max(10000, int(len(mesh.faces) * 0.5))

print(f"Reducing to {target_faces} faces...")
mesh = mesh.simplify_quadric_decimation(target_faces)

mesh.export("models/small_model.stl")
