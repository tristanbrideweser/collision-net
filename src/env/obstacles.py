# src/env/obstacles.py

import random 
import pybullet as p

def spawn_clutter(num_objects, object_types, size_min, size_max, base_mass):
    obj_ids = []
    for _ in range(num_objects):
        shape_type = random.choice(object_types)
        s = random.uniform(size_min, size_max)
        color = [random.random(), random.random(), random.random(), 1]
        
        # Initialize args
        col_args = {"shapeType": shape_type}
        vis_args = {"shapeType": shape_type, "rgbaColor": color}

        if shape_type == p.GEOM_BOX:
            # Boxes use halfExtents: [x, y, z]
            extents = [s, s, s]
            col_args["halfExtents"] = extents
            vis_args["halfExtents"] = extents
            
        elif shape_type == p.GEOM_CYLINDER:
            # Cylinders use radius and height (height = length)
            col_args["radius"] = s
            col_args["height"] = s * 2
            vis_args["radius"] = s
            vis_args["length"] = s * 2
            
        elif shape_type == p.GEOM_SPHERE:
            # Spheres only use radius
            col_args["radius"] = s
            vis_args["radius"] = s

        col = p.createCollisionShape(**col_args)
        vis = p.createVisualShape(**vis_args)
        
        # Position them on the table (ensure Z is high enough)
        pos = [random.uniform(0.3, 0.8), random.uniform(-0.3, 0.3), 0.65]
        
        obj_id = p.createMultiBody(baseMass=base_mass,
                                   baseCollisionShapeIndex=col,
                                   baseVisualShapeIndex=vis,
                                   basePosition=pos)
        obj_ids.append(obj_id)
    return obj_ids

def spawn_stack(base_pos, num_cubes, cube_size, mass=0.1):
    """
    Spawns a vertical stack of cubes at a specific (x, y) location.
    base_pos: [x, y]
    cube_size: float (half-extent, so 0.02 means a 4cm cube)
    """
    obj_ids = []
    # Start Z at table height (0.625) + half-extent of the first cube
    table_top_z = 0.625 + 0.05 # 0.625 is your Panda base height, table is 0.05 thick
    
    # We use a loop to increment the Z height for each cube
    for i in range(num_cubes):
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[cube_size]*3)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[cube_size]*3,
                                  rgbaColor=[0.8, 0.2, 0.2, 1]) # Red stack
        
        # Calculate Z: base + (layer_index * full_height)
        # Adding a tiny 0.001 gap prevents the physics engine from "exploding"
        z_pos = table_top_z + (i * (cube_size * 2 + 0.001)) + cube_size
        
        pos = [base_pos[0], base_pos[1], z_pos]
        
        obj_id = p.createMultiBody(baseMass=mass,
                                   baseCollisionShapeIndex=col,
                                   baseVisualShapeIndex=vis,
                                   basePosition=pos)
        obj_ids.append(obj_id)
        
    return obj_ids

def add_wall(position, size=(0.01, 0.3, 0.15)):
    """Add a vertical wall/barrier on the table."""
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(size))
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=list(size),
                              rgbaColor=[0.4, 0.4, 0.4, 0.8])
    wall_id = p.createMultiBody(baseMass=0,
                                baseCollisionShapeIndex=col,
                                baseVisualShapeIndex=vis,
                                basePosition=position)
    return wall_id

