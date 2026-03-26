"""
GDS/OAS layout file I/O for optical lithography simulator.
Uses gdstk library for reading GDS2 and OASIS format files.

Supports:
- GDS2 (.gds) via gdstk.read_gds()
- OASIS (.oas) via gdstk.read_oas()
- Cell hierarchy flattening
- Polygon extraction by layer
- Coordinate conversion: GDS units → simulation grid
- Progress callbacks for async UI integration
- Max display polygon sampling to limit renderer load
"""
import os
import random
import numpy as np
from typing import List, Optional, Dict, Tuple, Callable

from dataclasses import dataclass, field

try:
    import gdstk
    HAS_GDSTK = True
except ImportError:
    HAS_GDSTK = False


@dataclass
class BoundingBox:
    """Bounding box in nm coordinates."""
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def center(self) -> Tuple[float, float]:
        return (0.5*(self.xmin + self.xmax), 0.5*(self.ymin + self.ymax))


@dataclass
class LayerInfo:
    """Information about a GDS layer."""
    layer: int
    datatype: int
    n_polygons: int
    bbox: Optional[BoundingBox] = None
    color: Tuple[float, float, float] = (0.5, 0.5, 0.8)


@dataclass
class LayoutData:
    """
    Container for layout data extracted from GDS/OAS files.
    Stores polygons organized by layer number.
    All coordinates in nm.
    """
    filepath: str
    format: str                    # 'gds' or 'oas'
    unit_nm: float                 # Layout database unit in nm
    precision_nm: float            # Layout precision in nm
    top_cell_name: str
    layers: Dict[int, LayerInfo]   # layer_num -> LayerInfo
    polygons_by_layer: Dict[int, List[np.ndarray]]  # layer_num -> list of (N,2) arrays in nm
    bounding_box: Optional[BoundingBox] = None

    def get_layer_numbers(self) -> List[int]:
        return sorted(self.layers.keys())

    def get_polygons(self, layer: int) -> List[np.ndarray]:
        return self.polygons_by_layer.get(layer, [])

    def get_all_polygons(self) -> List[np.ndarray]:
        all_polys = []
        for polys in self.polygons_by_layer.values():
            all_polys.extend(polys)
        return all_polys


class LayoutReader:
    """
    Read GDS2 and OASIS layout files using gdstk.
    Returns LayoutData with polygons in nm units.
    """

    def __init__(self):
        if not HAS_GDSTK:
            import warnings
            warnings.warn("gdstk not installed. Layout I/O will use mock data.")

    def read_gds(self, filepath: str, top_cell: Optional[str] = None,
                 on_progress: Optional[Callable[[str, int], None]] = None,
                 max_display_polygons: int = 50000) -> 'LayoutData':
        """
        Read GDS2 file.
        Args:
            filepath: Path to .gds file
            top_cell: Top cell name (auto-detected if None)
            on_progress: Optional callback(step: str, percent: int)
            max_display_polygons: Max polygons to store per layer for display
        Returns:
            LayoutData with all polygons in nm
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError("GDS file not found: {}".format(filepath))

        if on_progress:
            on_progress("Reading file...", 5)

        if not HAS_GDSTK:
            if on_progress:
                on_progress("Done", 100)
            return self._create_mock_layout(filepath, 'gds')

        lib = gdstk.read_gds(filepath)
        return self._process_library(lib, filepath, 'gds', top_cell,
                                     on_progress=on_progress,
                                     max_display_polygons=max_display_polygons)

    def read_oas(self, filepath: str, top_cell: Optional[str] = None,
                 on_progress: Optional[Callable[[str, int], None]] = None,
                 max_display_polygons: int = 50000) -> 'LayoutData':
        """
        Read OASIS file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError("OAS file not found: {}".format(filepath))

        if on_progress:
            on_progress("Reading file...", 5)

        if not HAS_GDSTK:
            if on_progress:
                on_progress("Done", 100)
            return self._create_mock_layout(filepath, 'oas')

        lib = gdstk.read_oas(filepath)
        return self._process_library(lib, filepath, 'oas', top_cell,
                                     on_progress=on_progress,
                                     max_display_polygons=max_display_polygons)

    def read(self, filepath: str, top_cell: Optional[str] = None,
             on_progress: Optional[Callable[[str, int], None]] = None,
             max_display_polygons: int = 50000) -> 'LayoutData':
        """Auto-detect format from file extension."""
        ext = os.path.splitext(filepath)[1].lower()
        if ext in ('.gds', '.gds2'):
            return self.read_gds(filepath, top_cell,
                                 on_progress=on_progress,
                                 max_display_polygons=max_display_polygons)
        elif ext in ('.oas', '.oasis'):
            return self.read_oas(filepath, top_cell,
                                 on_progress=on_progress,
                                 max_display_polygons=max_display_polygons)
        else:
            # Try GDS first, then OAS
            try:
                return self.read_gds(filepath, top_cell,
                                     on_progress=on_progress,
                                     max_display_polygons=max_display_polygons)
            except Exception:
                return self.read_oas(filepath, top_cell,
                                     on_progress=on_progress,
                                     max_display_polygons=max_display_polygons)

    def _process_library(self, lib, filepath: str, fmt: str,
                          top_cell: Optional[str],
                          on_progress: Optional[Callable[[str, int], None]] = None,
                          max_display_polygons: int = 50000) -> 'LayoutData':
        """Process gdstk Library into LayoutData."""
        # Get units (convert to nm)
        unit = getattr(lib, 'unit', 1e-6)   # Default: 1 micron
        precision = getattr(lib, 'precision', 1e-9)

        # Convert to nm scale factor
        unit_nm = unit * 1e9       # meters -> nm
        precision_nm = precision * 1e9

        # Find top cell
        cells = lib.cells
        if not cells:
            if on_progress:
                on_progress("Done", 100)
            return self._create_mock_layout(filepath, fmt)

        if top_cell:
            cell = next((c for c in cells if c.name == top_cell), cells[0])
        else:
            # Use first top-level cell (no references to it from others)
            referenced = set()
            for c in cells:
                for ref in c.references:
                    if hasattr(ref, 'cell'):
                        referenced.add(ref.cell.name)
            top_cells = [c for c in cells if c.name not in referenced]
            cell = top_cells[0] if top_cells else cells[0]

        top_cell_name = cell.name

        # Flatten hierarchy
        if on_progress:
            on_progress("Flattening hierarchy...", 20)

        try:
            flat_cell = cell.copy(name=top_cell_name + '_flat')
            flat_cell.flatten()
        except Exception:
            flat_cell = cell

        # Extract polygons - collect all first to know total count for progress
        if on_progress:
            on_progress("Extracting polygons...", 40)

        raw_by_layer: Dict[int, List[np.ndarray]] = {}
        all_flat_polys = flat_cell.polygons
        total = len(all_flat_polys)

        for idx, poly in enumerate(all_flat_polys):
            layer_num = poly.layer
            pts = np.array(poly.points) * unit_nm  # Convert to nm

            if layer_num not in raw_by_layer:
                raw_by_layer[layer_num] = []
            raw_by_layer[layer_num].append(pts)

            # Emit per-polygon progress from 40 -> 80
            if on_progress and total > 0 and idx % max(1, total // 20) == 0:
                pct = 40 + int(40 * idx / total)
                on_progress("Extracting polygons...", pct)

        # Apply per-layer budget sampling and compute layer info
        if on_progress:
            on_progress("Computing bounding boxes...", 90)

        n_layers = len(raw_by_layer)
        polygons_by_layer: Dict[int, List[np.ndarray]] = {}
        layers: Dict[int, 'LayerInfo'] = {}
        all_coords = []

        layer_keys = sorted(raw_by_layer.keys())
        per_layer_budget = max(1, max_display_polygons // max(1, n_layers))

        for li, layer_num in enumerate(layer_keys):
            polys = raw_by_layer[layer_num]
            real_count = len(polys)

            # Sample if over budget
            if real_count > per_layer_budget:
                sampled = random.sample(polys, per_layer_budget)
            else:
                sampled = polys

            polygons_by_layer[layer_num] = sampled

            layer_coords = np.vstack(polys)  # Use all polys for bbox accuracy
            all_coords.append(layer_coords)
            bbox = BoundingBox(
                xmin=float(np.min(layer_coords[:, 0])),
                ymin=float(np.min(layer_coords[:, 1])),
                xmax=float(np.max(layer_coords[:, 0])),
                ymax=float(np.max(layer_coords[:, 1]))
            )
            layers[layer_num] = LayerInfo(
                layer=layer_num,
                datatype=0,
                n_polygons=real_count,  # Original count, not sampled
                bbox=bbox
            )

        # Overall bounding box
        if all_coords:
            all_pts = np.vstack(all_coords)
            bbox = BoundingBox(
                xmin=float(np.min(all_pts[:, 0])),
                ymin=float(np.min(all_pts[:, 1])),
                xmax=float(np.max(all_pts[:, 0])),
                ymax=float(np.max(all_pts[:, 1]))
            )
        else:
            bbox = BoundingBox(0, 0, 1000, 1000)

        if on_progress:
            on_progress("Done", 100)

        return LayoutData(
            filepath=filepath,
            format=fmt,
            unit_nm=unit_nm,
            precision_nm=precision_nm,
            top_cell_name=top_cell_name,
            layers=layers,
            polygons_by_layer=polygons_by_layer,
            bounding_box=bbox
        )

    def _create_mock_layout(self, filepath: str, fmt: str) -> 'LayoutData':
        """Create mock layout data for testing when gdstk unavailable."""
        # Simple line/space pattern
        N_polys = 5
        polygons = []
        for i in range(N_polys):
            x0 = i * 400.0
            poly = np.array([
                [x0, 0.0], [x0+200.0, 0.0],
                [x0+200.0, 1000.0], [x0, 1000.0]
            ])
            polygons.append(poly)

        layer1 = LayerInfo(layer=1, datatype=0, n_polygons=N_polys)
        bbox = BoundingBox(0, 0, 2000, 1000)
        layer1.bbox = bbox

        return LayoutData(
            filepath=filepath, format=fmt,
            unit_nm=1000.0, precision_nm=1.0,
            top_cell_name='MOCK',
            layers={1: layer1},
            polygons_by_layer={1: polygons},
            bounding_box=bbox
        )


class MaskGridGenerator:
    """
    Convert GDS layout polygons to simulation mask grids.
    Handles coordinate transforms between layout and simulation domains.
    """

    def __init__(self, grid_size: int, domain_size_nm: float):
        self.grid_size = grid_size
        self.domain_size_nm = domain_size_nm
        self.dx_nm = domain_size_nm / grid_size

    def polygons_to_grid(self, polygons: List[np.ndarray],
                          origin_nm: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        Rasterize list of polygon coordinate arrays to binary grid.

        Args:
            polygons: List of (N,2) arrays in nm
            origin_nm: (x0, y0) lower-left corner of simulation domain in nm
                       (None = auto-detect from polygon bounds)
        Returns:
            (grid_size, grid_size) float array, 1.0=clear, 0.0=opaque
        """
        N = self.grid_size

        if origin_nm is None and polygons:
            # Auto-detect origin from bounding box
            all_pts = np.vstack(polygons)
            origin_nm = (float(np.min(all_pts[:, 0])), float(np.min(all_pts[:, 1])))
        elif origin_nm is None:
            origin_nm = (0.0, 0.0)

        ox, oy = origin_nm
        dx = self.dx_nm

        grid = np.zeros((N, N), dtype=np.float64)

        for poly_nm in polygons:
            if len(poly_nm) < 3:
                continue
            # Translate to grid coordinates
            poly_grid = np.column_stack([
                (poly_nm[:, 0] - ox) / dx,
                (poly_nm[:, 1] - oy) / dx
            ])
            fill_mask = self._rasterize_polygon(poly_grid, N)
            grid[fill_mask] = 1.0

        return grid

    def _rasterize_polygon(self, poly_grid: np.ndarray, N: int) -> np.ndarray:
        """Rasterize polygon in grid coordinates to boolean mask."""
        try:
            from matplotlib.path import Path
            path = Path(poly_grid)
            xi = np.arange(N)
            yi = np.arange(N)
            XI, YI = np.meshgrid(xi, yi, indexing='ij')
            pts = np.column_stack([XI.ravel(), YI.ravel()])
            mask = path.contains_points(pts).reshape(N, N)
            return mask
        except Exception:
            # Fallback bounding box
            xmin = max(0, int(np.min(poly_grid[:, 0])))
            xmax = min(N-1, int(np.max(poly_grid[:, 0])))
            ymin = max(0, int(np.min(poly_grid[:, 1])))
            ymax = min(N-1, int(np.max(poly_grid[:, 1])))
            mask = np.zeros((N, N), dtype=bool)
            mask[xmin:xmax+1, ymin:ymax+1] = True
            return mask

    def get_simulation_domain(self, layout: 'LayoutData',
                               center_nm: Optional[Tuple[float, float]] = None,
                               layers: Optional[List[int]] = None) -> np.ndarray:
        """
        Extract simulation domain from layout centered at center_nm.

        Args:
            layout: LayoutData from LayoutReader
            center_nm: Center of simulation domain (None = layout center)
            layers: Layer numbers to include (None = all layers)
        Returns:
            Binary grid (1=clear, 0=opaque)
        """
        if center_nm is None and layout.bounding_box:
            center_nm = layout.bounding_box.center
        elif center_nm is None:
            center_nm = (0.0, 0.0)

        cx, cy = center_nm
        half = 0.5 * self.domain_size_nm
        origin_nm = (cx - half, cy - half)

        # Collect polygons from selected layers
        if layers is None:
            layers = layout.get_layer_numbers()

        all_polygons = []
        for layer_num in layers:
            all_polygons.extend(layout.get_polygons(layer_num))

        if not all_polygons:
            return np.ones((self.grid_size, self.grid_size), dtype=np.float64)

        return self.polygons_to_grid(all_polygons, origin_nm)

    def apply_bias(self, grid: np.ndarray, bias_nm: float) -> np.ndarray:
        """
        Apply CD bias to binary mask grid.
        Positive bias expands clear features.
        """
        from scipy import ndimage

        if bias_nm == 0:
            return grid.copy()

        n_pixels = abs(bias_nm) / self.dx_nm
        n_pixels = max(1, int(round(n_pixels)))
        struct = ndimage.generate_binary_structure(2, 1)

        binary = grid > 0.5
        if bias_nm > 0:
            result = ndimage.binary_dilation(binary, struct, iterations=n_pixels)
        else:
            result = ndimage.binary_erosion(binary, struct, iterations=n_pixels)

        return result.astype(np.float64)


def read_layout(filepath: str, top_cell: Optional[str] = None) -> LayoutData:
    """Convenience function to read any layout file."""
    reader = LayoutReader()
    return reader.read(filepath, top_cell)


def layout_to_mask_grid(layout: LayoutData, grid_size: int,
                         domain_size_nm: float,
                         center_nm: Optional[Tuple[float, float]] = None,
                         layers: Optional[List[int]] = None) -> np.ndarray:
    """
    Convenience function: read layout and convert to mask grid.
    Returns binary mask array (1=clear, 0=opaque).
    """
    gen = MaskGridGenerator(grid_size, domain_size_nm)
    return gen.get_simulation_domain(layout, center_nm, layers)
