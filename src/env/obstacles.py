# src/env/obstacles.py

"""
obstacles.py — Workspace obstacles for learning-based collision detection.

Design philosophy:
  The goal is to create C-space collision boundaries that are complex and
  nonlinear. This means narrow passages, enclosures the arm must thread
  through, and layered barriers at multiple heights. Random scatter alone
  doesn't achieve this — the Panda's 7-DOF redundancy lets it route around
  small isolated objects trivially.

  Each difficulty level is defined by PASSAGE WIDTH, not just object count.
  Tighter gaps = exponentially harder C-space topology.

Difficulty targets (random config collision rate):
  sparse:       ~10-15%  (single barrier + light clutter)
  interspersed: ~20-30%  (multiple barriers + enclosures)
  dense:        ~35-50%  (layered barriers + tight passages + enclosures)
"""

import random
import numpy as np
import pybullet as p


# ──────────────────────────────────────────────
# Exclusion zone helper
# ──────────────────────────────────────────────

def _in_exclusion_zone(x, y, exclusion_zones, clearance=0.15):
    """
    Check if (x, y) is too close to any exclusion zone.

    Args:
        x, y: candidate position
        exclusion_zones: list of [x, y, z] positions to avoid
        clearance: minimum XY distance from any zone center

    Returns:
        True if the position should be rejected
    """
    for zone in exclusion_zones:
        if np.hypot(x - zone[0], y - zone[1]) < clearance:
            return True
    return False


# ──────────────────────────────────────────────
# Difficulty parameters
# ──────────────────────────────────────────────

DIFFICULTY_PARAMS = {
    "sparse": {
        "num_barriers": 1,
        "gap_size": 0.16,              # was 0.14 — wider gap, clearly easier
        "num_enclosures": 0,
        "enclosure_opening": 0.16,
        "num_columns": 2,              # was 3
        "column_radius_range": (0.03, 0.06),  # was (0.04, 0.07) — thinner
        "num_shelves": 0,
        "num_scatter": 2,              # was 3
        "scatter_size_range": (0.02, 0.06),
    },
    "interspersed": {
        "num_barriers": 2,
        "gap_size": 0.09,              # was 0.10
        "num_enclosures": 1,
        "enclosure_opening": 0.12,
        "num_columns": 6,              # was 5
        "column_radius_range": (0.05, 0.09),  # was (0.04, 0.08)
        "num_shelves": 1,
        "num_scatter": 6,              # was 5
        "scatter_size_range": (0.04, 0.10),
    },
       "dense": {
        "num_barriers": 3,
        "gap_size": 0.06,
        "num_enclosures": 2,
        "enclosure_opening": 0.09,
        "num_columns": 12,             # was 10
        "column_radius_range": (0.07, 0.12),  # was (0.06, 0.11)
        "num_shelves": 3,
        "num_scatter": 15,             # was 12
        "scatter_size_range": (0.05, 0.13),
    },
}


# ──────────────────────────────────────────────
# Primitive builders
# ──────────────────────────────────────────────

def _make_box(position, half_extents, color, mass=0.0):
    """Create a static box and return its body ID."""
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(half_extents))
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=list(half_extents),
                              rgbaColor=color)
    return p.createMultiBody(baseMass=mass,
                             baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis,
                             basePosition=list(position))


def _make_cylinder(position, radius, height, color, mass=0.0):
    """Create a static cylinder and return its body ID."""
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height,
                              rgbaColor=color)
    return p.createMultiBody(baseMass=mass,
                             baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis,
                             basePosition=list(position))


# ──────────────────────────────────────────────
# Barrier wall with controlled gap
# ──────────────────────────────────────────────

def spawn_barrier_wall(wall_x, gap_center_y, gap_size,
                       wall_y_range=(-0.35, 0.35),
                       wall_height=0.30, table_z=0.65):
    """
    A vertical wall at x=wall_x spanning the Y range, with a single gap
    of width `gap_size` centered at `gap_center_y`.

    The gap forces the arm to thread through a narrow passage — this is
    where the interesting C-space topology lives.

    Includes a top cap and bottom lip so the arm can't go over or under.
    """
    obj_ids = []
    y_min, y_max = wall_y_range
    gap_lo = gap_center_y - gap_size / 2
    gap_hi = gap_center_y + gap_size / 2

    # Left section (y_min to gap_lo)
    if gap_lo > y_min + 0.02:
        center_y = (y_min + gap_lo) / 2
        half_len = (gap_lo - y_min) / 2
        obj_ids.append(_make_box(
            position=[wall_x, center_y, table_z + wall_height / 2],
            half_extents=[0.012, half_len, wall_height / 2],
            color=[0.55, 0.55, 0.55, 0.92],
        ))

    # Right section (gap_hi to y_max)
    if gap_hi < y_max - 0.02:
        center_y = (gap_hi + y_max) / 2
        half_len = (y_max - gap_hi) / 2
        obj_ids.append(_make_box(
            position=[wall_x, center_y, table_z + wall_height / 2],
            half_extents=[0.012, half_len, wall_height / 2],
            color=[0.55, 0.55, 0.55, 0.92],
        ))

    # Top cap above the gap (prevents going over)
    cap_height = 0.12
    obj_ids.append(_make_box(
        position=[wall_x, gap_center_y, table_z + wall_height + cap_height / 2],
        half_extents=[0.012, gap_size / 2 + 0.02, cap_height / 2],
        color=[0.55, 0.55, 0.55, 0.92],
    ))

    # Bottom lip under the gap (prevents going under)
    lip_height = 0.06
    obj_ids.append(_make_box(
        position=[wall_x, gap_center_y, table_z + lip_height / 2],
        half_extents=[0.012, gap_size / 2 + 0.02, lip_height / 2],
        color=[0.55, 0.55, 0.55, 0.92],
    ))

    return obj_ids


# ──────────────────────────────────────────────
# Enclosure (3-sided box with narrow opening)
# ──────────────────────────────────────────────

def spawn_enclosure(center, width=0.28, depth=0.20, height=0.25,
                    opening_width=None, table_z=0.65):
    """
    A box open on one side with a controlled opening width.
    The arm must enter through the opening — creating a concave
    C-space obstacle that's hard for geometric planners.

    Args:
        center: [x, y] position on the table
        opening_width: width of the front opening. If None, full width.
                       If < width, partial front walls narrow the entrance.
    """
    obj_ids = []
    t = 0.012  # wall thickness
    bz = table_z
    cx, cy = center

    # Back wall
    obj_ids.append(_make_box(
        position=[cx - depth / 2, cy, bz + height / 2],
        half_extents=[t, width / 2, height / 2],
        color=[0.3, 0.35, 0.6, 0.85],
    ))

    # Left wall
    obj_ids.append(_make_box(
        position=[cx, cy - width / 2, bz + height / 2],
        half_extents=[depth / 2, t, height / 2],
        color=[0.3, 0.35, 0.6, 0.85],
    ))

    # Right wall
    obj_ids.append(_make_box(
        position=[cx, cy + width / 2, bz + height / 2],
        half_extents=[depth / 2, t, height / 2],
        color=[0.3, 0.35, 0.6, 0.85],
    ))

    # Ceiling
    obj_ids.append(_make_box(
        position=[cx, cy, bz + height],
        half_extents=[depth / 2, width / 2, t],
        color=[0.3, 0.35, 0.6, 0.85],
    ))

    # Partial front walls to narrow the opening
    if opening_width is not None and opening_width < width:
        flap = (width - opening_width) / 2
        if flap > 0.02:
            # Left flap
            obj_ids.append(_make_box(
                position=[cx + depth / 2, cy - width / 2 + flap / 2, bz + height / 2],
                half_extents=[t, flap / 2, height / 2],
                color=[0.3, 0.35, 0.6, 0.85],
            ))
            # Right flap
            obj_ids.append(_make_box(
                position=[cx + depth / 2, cy + width / 2 - flap / 2, bz + height / 2],
                half_extents=[t, flap / 2, height / 2],
                color=[0.3, 0.35, 0.6, 0.85],
            ))

    return obj_ids


# ──────────────────────────────────────────────
# Horizontal shelf barrier
# ──────────────────────────────────────────────

def spawn_shelf(center_x, center_y, shelf_z, span_y=0.25, depth_x=0.12):
    """
    A flat horizontal plate that blocks vertical arm movement.
    Forces the arm to plan around it in the Z dimension.
    Includes small support columns for visual clarity.
    """
    obj_ids = []
    t = 0.008

    # Shelf surface
    obj_ids.append(_make_box(
        position=[center_x, center_y, shelf_z],
        half_extents=[depth_x / 2, span_y / 2, t],
        color=[0.45, 0.35, 0.25, 0.9],
    ))

    # Support columns
    col_h = shelf_z - 0.65
    if col_h > 0.03:
        for dy in [-span_y / 2 + 0.01, span_y / 2 - 0.01]:
            obj_ids.append(_make_cylinder(
                position=[center_x, center_y + dy, 0.65 + col_h / 2],
                radius=0.01,
                height=col_h,
                color=[0.4, 0.3, 0.2, 0.9],
            ))

    return obj_ids


# ──────────────────────────────────────────────
# Thick columns
# ──────────────────────────────────────────────

def spawn_columns(num_columns, radius_range, x_range=(0.15, 0.65),
                  y_range=(-0.35, 0.35), table_z=0.65, exclusion_zones=None):
    """
    Tall cylinders in the workspace. Sized to actually block the
    swept volume of the arm's links, not just the end-effector.
    Enforces minimum spacing between columns and respects exclusion zones.
    """
    obj_ids = []
    positions = []
    zones = exclusion_zones or []

    for _ in range(num_columns):
        placed = False
        for _attempt in range(50):
            cx = random.uniform(*x_range)
            cy = random.uniform(*y_range)
            if (all(np.hypot(cx - px, cy - py) > 0.12 for px, py in positions)
                    and not _in_exclusion_zone(cx, cy, zones)):
                placed = True
                break

        if not placed:
            continue

        positions.append((cx, cy))
        radius = random.uniform(*radius_range)
        height = random.uniform(0.20, 0.45)

        obj_ids.append(_make_cylinder(
            position=[cx, cy, table_z + height / 2],
            radius=radius,
            height=height,
            color=[random.uniform(0.3, 0.7), random.uniform(0.3, 0.7),
                   random.uniform(0.3, 0.7), 1.0],
        ))

    return obj_ids


# ──────────────────────────────────────────────
# Scattered clutter (supplemental)
# ──────────────────────────────────────────────

def spawn_scatter(num_objects, size_range, x_range=(0.10, 0.65),
                  y_range=(-0.35, 0.35), table_z=0.65, exclusion_zones=None):
    """Small random objects for local complexity around the main obstacles."""
    obj_ids = []
    zones = exclusion_zones or []

    for _ in range(num_objects):
        # Find a valid position
        pos = None
        for _attempt in range(50):
            candidate = [random.uniform(*x_range),
                         random.uniform(*y_range),
                         table_z + random.uniform(0.02, 0.25)]
            if not _in_exclusion_zone(candidate[0], candidate[1], zones):
                pos = candidate
                break
        if pos is None:
            continue

        shape = random.choice([p.GEOM_BOX, p.GEOM_SPHERE, p.GEOM_CYLINDER])
        s = random.uniform(*size_range)
        color = [random.random(), random.random(), random.random(), 1.0]

        if shape == p.GEOM_BOX:
            obj_ids.append(_make_box(pos, [s, s, s], color))
        elif shape == p.GEOM_SPHERE:
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=s)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=s, rgbaColor=color)
            obj_ids.append(p.createMultiBody(
                baseMass=0, baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis, basePosition=pos))
        else:
            obj_ids.append(_make_cylinder(pos, s, s * 3, color))

    return obj_ids


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────

def spawn_structured_clutter(num_objects, robot_id, start_pos, goal_pos,
                             difficulty="sparse", table_z=0.65,
                             exclusion_points=None):
    """
    Spawn a constrained workspace scene for the given difficulty.

    The `num_objects` parameter is accepted for API compatibility with
    main.py but is NOT used — obstacle count and sizing are controlled
    entirely by DIFFICULTY_PARAMS to ensure meaningful C-space topology.

    Args:
        num_objects: (unused, kept for interface compatibility)
        robot_id: PyBullet body ID of the Panda
        start_pos: EE position at start config [x, y, z]
        goal_pos: EE position at goal config [x, y, z]
        difficulty: "sparse", "interspersed", or "dense"
        table_z: height of the table surface
        exclusion_points: list of [x, y, z] positions to keep clear.
                          Typically all link positions for start + goal configs.
                          If None, falls back to start_pos/goal_pos only.

    Returns:
        list of PyBullet body IDs for all spawned obstacles
    """
    params = DIFFICULTY_PARAMS[difficulty]
    obj_ids = []

    ee_exclusion = []
    if start_pos is not None:
        ee_exclusion.append(start_pos)
    if goal_pos is not None:
        ee_exclusion.append(goal_pos)

    # Build exclusion zones — use full link positions if provided,
    # otherwise fall back to just the EE positions
    if exclusion_points is not None:
        exclusion_zones = exclusion_points
    else:
        exclusion_zones = []
        if start_pos is not None:
            exclusion_zones.append(start_pos)
        if goal_pos is not None:
            exclusion_zones.append(goal_pos)

    # Midpoint Y between start and goal — bias barrier gaps here
    # so the arm always has a feasible passage between endpoints
    if start_pos is not None and goal_pos is not None:
        mid_y = (start_pos[1] + goal_pos[1]) / 2
    else:
        mid_y = 0.0

    # ── Layer 1: Barrier walls with controlled gaps ──────────────
    #
    # Gap is biased toward mid_y (between start and goal) with some
    # randomness, so there's always a feasible passage but it's not
    # trivially straight.
    num_barriers = params["num_barriers"]
    if num_barriers > 0:
        barrier_xs = np.linspace(0.25, 0.55, num_barriers + 2)[1:-1]
        for bx in barrier_xs:
            # Skip barrier if it's at the same X as start or goal
            skip = False
            for zone in ee_exclusion:
                if abs(bx - zone[0]) < 0.10:
                    skip = True
                    break
            if skip:
                continue

            # Bias gap toward midpoint with jitter
            gap_y = mid_y + random.uniform(-0.08, 0.08)
            gap_y = np.clip(gap_y, -0.20, 0.20)
            wall_h = random.uniform(0.25, 0.38)

            obj_ids.extend(spawn_barrier_wall(
                wall_x=bx,
                gap_center_y=gap_y,
                gap_size=params["gap_size"],
                wall_height=wall_h,
                table_z=table_z,
            ))

    # ── Layer 2: Enclosures ──────────────────────────────────────
    #
    # Only place enclosures that don't overlap with start/goal.
    num_enc = params["num_enclosures"]
    enc_candidates = [
        [0.55, -0.15],
        [0.50, 0.20],
        [0.45, -0.25],
        [0.40, 0.25],
    ]
    enc_placed = 0
    for ec in enc_candidates:
        if enc_placed >= num_enc:
            break
        if _in_exclusion_zone(ec[0], ec[1], exclusion_zones, clearance=0.18):
            continue
        obj_ids.extend(spawn_enclosure(
            center=ec,
            width=random.uniform(0.22, 0.30),
            depth=random.uniform(0.16, 0.22),
            height=random.uniform(0.20, 0.30),
            opening_width=params["enclosure_opening"],
            table_z=table_z,
        ))
        enc_placed += 1

    # ── Layer 3: Horizontal shelves ──────────────────────────────
    for _ in range(params["num_shelves"]):
        for _attempt in range(30):
            sx = random.uniform(0.25, 0.55)
            sy = random.uniform(-0.15, 0.15)
            if not _in_exclusion_zone(sx, sy, exclusion_zones):
                sz = table_z + random.uniform(0.12, 0.32)
                obj_ids.extend(spawn_shelf(sx, sy, sz))
                break

    # ── Layer 4: Thick columns ───────────────────────────────────
    obj_ids.extend(spawn_columns(
        num_columns=params["num_columns"],
        radius_range=params["column_radius_range"],
        table_z=table_z,
        exclusion_zones=exclusion_zones,
    ))

    # ── Layer 5: Scattered clutter ───────────────────────────────
    obj_ids.extend(spawn_scatter(
        num_objects=params["num_scatter"],
        size_range=params["scatter_size_range"],
        table_z=table_z,
        exclusion_zones=exclusion_zones,
    ))

    return obj_ids


# ──────────────────────────────────────────────
# Legacy functions (kept for A/B comparison)
# ──────────────────────────────────────────────

def spawn_clutter(num_objects, object_types, size_min, size_max, base_mass,
                  robot_id, start_pos, goal_pos,
                  x_range, y_range, clearance=0.15, table_z=0.65):
    """Original random scatter — kept for baseline comparison."""
    obj_ids = []
    spawned_positions = []
    robot_base_pos, _ = p.getBasePositionAndOrientation(robot_id)

    for _ in range(num_objects):
        valid_pos = False
        attempts = 0
        while not valid_pos and attempts < 100:
            pos = [random.uniform(*x_range), random.uniform(*y_range),
                   table_z + random.uniform(0.05, 0.35)]
            dist_to_robot = np.linalg.norm(np.array(pos) - np.array(robot_base_pos))
            dist_to_start = np.linalg.norm(np.array(pos) - np.array(start_pos))
            dist_to_goal = np.linalg.norm(np.array(pos) - np.array(goal_pos))
            too_close = any(
                np.linalg.norm(np.array(pos) - np.array(sp)) < clearance
                for sp in spawned_positions)
            if (all(d > clearance for d in [dist_to_robot, dist_to_start, dist_to_goal])
                    and not too_close):
                valid_pos = True
            attempts += 1

        if valid_pos:
            shape_type = random.choice(object_types)
            s = random.uniform(size_min, size_max)
            color = [random.random(), random.random(), random.random(), 1]
            col_args = {"shapeType": shape_type}
            vis_args = {"shapeType": shape_type, "rgbaColor": color}
            if shape_type == p.GEOM_BOX:
                col_args["halfExtents"] = vis_args["halfExtents"] = [s, s, s]
            elif shape_type == p.GEOM_CYLINDER:
                col_args["radius"] = vis_args["radius"] = s
                col_args["height"] = s * 2
                vis_args["length"] = s * 2
            elif shape_type == p.GEOM_SPHERE:
                col_args["radius"] = vis_args["radius"] = s
            col = p.createCollisionShape(**col_args)
            vis = p.createVisualShape(**vis_args)
            obj_id = p.createMultiBody(baseMass=base_mass,
                                       baseCollisionShapeIndex=col,
                                       baseVisualShapeIndex=vis,
                                       basePosition=pos)
            obj_ids.append(obj_id)
            spawned_positions.append(pos)
    return obj_ids


def spawn_stack(base_pos, num_cubes, cube_size, mass=0.0):
    """Spawns a vertical stack of cubes."""
    obj_ids = []
    table_top_z = 0.625 + 0.05
    for i in range(num_cubes):
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[cube_size] * 3)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[cube_size] * 3,
                                  rgbaColor=[0.8, 0.2, 0.2, 1])
        z_pos = table_top_z + (i * (cube_size * 2 + 0.001)) + cube_size
        obj_id = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col,
                                   baseVisualShapeIndex=vis,
                                   basePosition=[base_pos[0], base_pos[1], z_pos])
        obj_ids.append(obj_id)
    return obj_ids


def add_wall(position, size=(0.01, 0.3, 0.15)):
    """Add a simple wall."""
    return _make_box(position, size, [0.4, 0.4, 0.4, 0.8])