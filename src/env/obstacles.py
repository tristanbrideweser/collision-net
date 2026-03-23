import random 
import pybullet as p

def spawn_clutter(
        num_objects,
        size_min,
        size_max,
        base_mass):
    """
    create env clutter objects
    """
    obj_ids = []

    for _ in range(num_objects):
        shape_type = random.choice([p.GEOM_BOX, p.GEOM_CYLINDER, p.GEOM_SPHERE])
        size = [random.uniform(size_min, size_max)] * 3
        col = p.createCollisionShape(shape_type, halfExtents=size)
        vis = p.createVisualShape(shape_type,
                                  halfExtents=size,
                                  rgbaColor=[random.random(),
                                             random.random(),
                                             random.random(),
                                             1]
                                    )
        pos = [random.uniform(0.3, 0.7), random.uniform(-0.3, 0.3), 0.65]
        obj_id = p.createMultiBody(baseMass=base_mass,
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

