"""Evaluate one build on saved cases; report CPU/CUDA mismatches and exact float64 volumes."""
import importlib.util, json, sys
import numpy as np, torch
from scipy.spatial import HalfspaceIntersection, ConvexHull



PLANES = [[0,1,2,3],[3,2,6,7],[0,1,5,4],[0,3,7,4],[1,5,6,2],[4,5,6,7]]

def halfspaces(box):
    box = box.astype(np.float64); c = box.mean(0); hs = []
    for f in PLANES:
        q = box[f]; n = np.cross(q[2] - q[0], q[3] - q[1]); n /= np.linalg.norm(n)
        if np.dot(c - q.mean(0), n) < 0: n = -n          # inward
        hs.append(np.concatenate([-n, [np.dot(n, q.mean(0))]]))  # -n.x + n.q0 <= 0  <=> n.(x-q0) >= 0
    return np.array(hs)

def exact(b1, b2):
    hs = np.vstack([halfspaces(b1), halfspaces(b2)])
    # interior point via LP-free heuristic: Chebyshev center
    from scipy.optimize import linprog
    A, bb = hs[:, :3], -hs[:, 3]; norms = np.linalg.norm(A, axis=1)
    res = linprog([0, 0, 0, -1], A_ub=np.hstack([A, norms[:, None]]), b_ub=bb, bounds=[(None, None)] * 3 + [(0, None)])
    scale = np.abs(np.vstack([b1, b2])).max()
    if not res.success or res.x[3] < 1e-6 * scale:
        return 0.0
    try:
        v = HalfspaceIntersection(hs, res.x[:3]).intersections
    except Exception:
        return float('nan')
    return ConvexHull(v).volume

