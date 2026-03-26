#!/usr/bin/env python3
"""
Generate a representative test OASIS layout file for optical lithography simulator.

Features:
  Layer 1 — 100nm line/space pattern (20 lines, 2µm height)
  Layer 2 — 100nm contact holes on 250nm pitch (10x8 array)
  Layer 3 — L-shaped feature (corner rounding test)
  Layer 4 — 50nm dense lines in a sub-cell, instantiated twice (tests hierarchy)

Usage:
    python3 tests/fixtures/make_test_oas.py
"""
import os
import sys
import numpy as np

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

try:
    import gdstk
except ImportError:
    print("gdstk not installed. Install with: pip install gdstk")
    sys.exit(1)


def create_test_oas(output_path: str) -> None:
    # GDS unit = 1 µm, precision = 1 nm
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    top = lib.new_cell('TEST_LITHO')

    # ------------------------------------------------------------------
    # Layer 1: 100 nm line / 100 nm space  (pitch = 200 nm = 0.2 µm)
    # ------------------------------------------------------------------
    line_w = 0.100   # 100 nm in µm
    pitch  = 0.200   # 200 nm pitch
    n_lines = 20
    height  = 2.0    # 2 µm
    x_start = -n_lines * pitch / 2.0

    for i in range(n_lines):
        x0 = x_start + i * pitch
        top.add(gdstk.rectangle(
            (x0, -height / 2), (x0 + line_w, height / 2),
            layer=1, datatype=0))

    # ------------------------------------------------------------------
    # Layer 2: 100 nm contact holes on 250 nm pitch (10 × 8 array)
    #          Approximated as 16-sided polygons.
    # ------------------------------------------------------------------
    hole_d = 0.100   # 100 nm diameter
    pitch2 = 0.250   # 250 nm pitch
    n_hx, n_hy = 10, 8
    cx_off = -n_hx * pitch2 / 2.0
    cy_off = -n_hy * pitch2 / 2.0 + 2.8   # offset above the L/S region

    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    r = hole_d / 2.0
    for ix in range(n_hx):
        for iy in range(n_hy):
            cx = cx_off + ix * pitch2 + r
            cy = cy_off + iy * pitch2 + r
            pts = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]
            top.add(gdstk.Polygon(pts, layer=2, datatype=0))

    # ------------------------------------------------------------------
    # Layer 3: L-shaped feature (corner rounding / OPC test)
    # ------------------------------------------------------------------
    l_pts = [
        (1.1, -0.5), (2.1, -0.5), (2.1,  0.0),
        (1.4,  0.0), (1.4,  0.5), (1.1,  0.5),
    ]
    top.add(gdstk.Polygon(l_pts, layer=3, datatype=0))

    # ------------------------------------------------------------------
    # Layer 4: 50 nm dense lines in a sub-cell → two cell references
    #          (tests cell.get_polygons(apply_repetitions=True))
    # ------------------------------------------------------------------
    dense_cell = lib.new_cell('DENSE_LS_50NM')
    dense_w     = 0.050   # 50 nm
    dense_pitch = 0.100   # 100 nm pitch
    n_dense     = 12

    for i in range(n_dense):
        x0 = i * dense_pitch
        dense_cell.add(gdstk.rectangle(
            (x0, 0), (x0 + dense_w, 0.400),
            layer=4, datatype=0))

    # Single reference
    top.add(gdstk.Reference(dense_cell, origin=(-0.6, -2.2)))

    # Array reference: 2 columns, 1 row, 1.5 µm column spacing
    top.add(gdstk.Reference(
        dense_cell, origin=(-0.6, -2.8),
        columns=2, rows=1, spacing=(1.5, 0)))

    # ------------------------------------------------------------------
    # Write and verify
    # ------------------------------------------------------------------
    lib.write_oas(output_path)
    print("Written: {}".format(output_path))

    # Verify round-trip read and hierarchy flattening
    lib2  = gdstk.read_oas(output_path)
    top2  = next(c for c in lib2.cells if c.name == 'TEST_LITHO')
    flat  = top2.get_polygons(apply_repetitions=True)
    layers = sorted({p.layer for p in flat})
    per_layer = {la: sum(1 for p in flat if p.layer == la) for la in layers}
    print("Total polygons (flattened): {}".format(len(flat)))
    print("Per layer: {}".format(per_layer))


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, 'test_litho.oas')
    create_test_oas(out_path)
