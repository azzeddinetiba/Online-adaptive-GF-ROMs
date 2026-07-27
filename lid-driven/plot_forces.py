from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree


INTERFACE_LOADS_PATH = (
	"/Users/azzeddinetiba/Desktop/fsi_lid_driven_cavity/"
	"fomData/train_dt03/12mu12/coSimData/load_data.npy"
)
INTERFACE_COORDS_PATH = (
	"/Users/azzeddinetiba/Desktop/fsi_lid_driven_cavity/"
	"fomData/train_dt03/12mu12/coSimData/coords_interf.npy"
)
VTK_PATH = (
	"/Users/azzeddinetiba/Desktop/trackedSurrogateExamples/lid_driven_cavity/"
	"vtk_output_lid_fsi_csd/Structure_0_1.vtk"
)
DEFAULT_TITLES = ["Force field 1", "Force field 2", "Force field 3", "Force field 4"]


def extract_force_snapshot(force_matrix: np.ndarray, snapshot: int) -> np.ndarray:
	if force_matrix.ndim != 2:
		raise ValueError(f"Expected a 2D force matrix, got shape {force_matrix.shape}")

	n_rows, n_snaps = force_matrix.shape
	if n_rows % 2 != 0:
		raise ValueError(
			f"Expected an even number of rows (Fx, Fy pairs), got {n_rows}"
		)

	if snapshot < 0:
		snapshot = n_snaps + snapshot
	if snapshot < 0 or snapshot >= n_snaps:
		raise IndexError(f"Snapshot index {snapshot} out of range [0, {n_snaps - 1}]")

	return force_matrix[:, snapshot]


def choose_snapshot_by_max_norm(force_matrix: np.ndarray) -> int:
	reshaped = force_matrix.reshape(-1, 2, force_matrix.shape[1])
	snap_norm = np.linalg.norm(reshaped, axis=1).mean(axis=0)
	return int(np.argmax(snap_norm))


def map_interface_coords_to_mesh(coords: np.ndarray, mesh: pv.DataSet) -> np.ndarray:
	if coords.ndim != 2 or coords.shape[1] < 2:
		raise ValueError(
			f"Expected coords array with shape (n_points, >=2), got {coords.shape}"
		)

	mesh_points = mesh.points.copy()
	mesh_xy = mesh_points[:, :2]
	coords_xy = coords[:, :2]
	_, ids = cKDTree(mesh_xy).query(coords_xy)
	return mesh_points[ids]


def plot_four_forces(
	loads_paths: Sequence[Path],
	coords_path: Path,
	vtk_path: Path,
	snapshot: int | None,
	output_path: Path | None,
	window_size: tuple[int, int],
	titles: Sequence[str],
	loads: np.ndarray,
	global_target_factor: float = 0.5,
	cmap:str = 'Wistia'
) -> None:
	assert loads_paths is not None or loads is not None
	if loads_paths is not None:
		if len(loads_paths) != 4:
			raise ValueError(f"Expected four force arrays, got {len(loads_paths)}")
	else:
		if len(loads) != 4:
			raise ValueError(f"Expected four force arrays, got {len(loads)}")
	if len(titles) != 4:
		raise ValueError(f"Expected four titles, got {len(titles)}")

	coords = np.load(coords_path)
	mesh = pv.read(vtk_path)
	mesh.points[:, 1] *= 35
	coords_3d = map_interface_coords_to_mesh(coords, mesh)

	if loads_paths is not None:
		force_matrices = [np.load(path) for path in loads_paths]
	else:
		force_matrices = loads

	selected_snapshots: list[int] = []
	panel_magnitudes: list[np.ndarray] = []
	panel_vectors: list[np.ndarray] = []
	panel_coords: list[np.ndarray] = []
	panel_factors: list[np.ndarray] = []

	global_target = global_target_factor

	for force_matrix in force_matrices:
		selected_snapshot = snapshot if snapshot is not None else choose_snapshot_by_max_norm(force_matrix)
		selected_snapshots.append(selected_snapshot)

		force_vector = extract_force_snapshot(force_matrix, selected_snapshot)
		fx = force_vector[0::2]
		fy = force_vector[1::2]

		n_interface = coords.shape[0]
		if fx.shape[0] != n_interface:
			raise ValueError(
				"Mismatch between interface points and force entries: "
				f"coords={n_interface}, force_pairs={fx.shape[0]}"
			)

		magnitude = np.hypot(fx, fy)
		vectors = np.column_stack([fx, fy, np.zeros(n_interface)])

		xmin, xmax, ymin, ymax, _, _ = mesh.bounds
		dx = xmax - xmin
		dy = ymax - ymin
		max_extent = max(dx, dy) if max(dx, dy) > 0.0 else 1.0

		max_force = float(magnitude.max())
		target_arrow_length = 0.12 * max_extent
		arrow_factor_ = global_target / max_force if max_force > 0.0 else 1.0

		panel_magnitudes.append(magnitude)
		panel_vectors.append(vectors)
		panel_coords.append(coords_3d)
		panel_factors.append(arrow_factor_)

	global_max = max(float(mag.max()) for mag in panel_magnitudes) if panel_magnitudes else 0.0
	clim = (0.0, global_max)

	plotter = pv.Plotter(
		shape=(2, 2),
		off_screen=output_path is not None,
		window_size=window_size,
	)
	plotter.set_background("white")

	for panel_idx, (magnitude, vectors, coords_panel, title, selected_snapshot, arrow_factor) in enumerate(
		zip(panel_magnitudes, panel_vectors, panel_coords, titles, selected_snapshots, panel_factors)
	):
		plotter.subplot(panel_idx // 2, panel_idx % 2)
		plotter.add_mesh(
			mesh,
			color="#F9C390",
			opacity=1.0,
			show_edges=True,
			edge_color="black",
			line_width=1.0,
		)

		interface_poly = pv.PolyData(coords_panel)
		interface_poly["force_vec"] = vectors
		interface_poly["force_mag"] = magnitude

		plotter.add_points(
			interface_poly,
			color="#1f77b4",
			render_points_as_spheres=True,
			point_size=10,
			label="Interface nodes",
		)

		arrowed = interface_poly.glyph(
			orient="force_vec",
			scale="force_mag",
			factor=global_target,
			geom=pv.Arrow(tip_length=0.2, tip_radius=0.06, shaft_radius=0.025),
		)
		plotter.add_mesh(
			arrowed,
			scalars="force_mag",
			cmap=cmap,
			smooth_shading=True,
			# clim=clim,
			scalar_bar_args={"title": "Force magnitude"},
			label="Force vectors",
			show_scalar_bar=False
		)
		plotter.add_text(
			f"{title}",
			position="upper_left",
			font_size=12,
			color="black",
		)
		plotter.view_xy()
		plotter.camera.parallel_projection = True
		plotter.reset_camera()
		plotter.camera.zoom(3.5)
		plotter.hide_axes()

	if output_path is not None:
		output_path.parent.mkdir(parents=True, exist_ok=True)
		plotter.screenshot(str(output_path), transparent_background=True)
		print(f"Saved figure to: {output_path}")
		plotter.close()
	else:
		plotter.show()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Plot interface force vectors (Fx, Fy interleaved rows) on top of the "
			"structural boundary using interface coordinates and VTK geometry."
		)
	)
	parser.add_argument(
		"--loads-paths",
		type=Path,
		nargs=4,
		default=[Path(INTERFACE_LOADS_PATH)] * 4,
		help="Four force-array paths to plot in a 2x2 panel layout.",
	)
	parser.add_argument("--coords-path", type=Path, default=Path(INTERFACE_COORDS_PATH))
	parser.add_argument("--vtk-path", type=Path, default=Path(VTK_PATH))
	parser.add_argument(
		"--snapshot",
		type=int,
		default=None,
		help=(
			"Snapshot index to plot. Use negative indices for reverse indexing. "
			"If omitted, the snapshot with largest mean interface force norm is used for each field."
		),
	)
	parser.add_argument(
		"--titles",
		nargs=4,
		default=DEFAULT_TITLES,
		help="Four titles for the subplots.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=None,
		help="Optional output path (e.g. forces_vectors.png). If omitted, opens a window.",
	)
	parser.add_argument(
		"--window-width",
		type=int,
		default=1800,
		help="Render window width in pixels.",
	)
	parser.add_argument(
		"--window-height",
		type=int,
		default=900,
		help="Render window height in pixels.",
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	plot_four_forces(
		loads_paths=args.loads_paths,
		coords_path=args.coords_path,
		vtk_path=args.vtk_path,
		snapshot=args.snapshot,
		output_path=args.output,
		window_size=(args.window_width, args.window_height),
		titles=args.titles,
		loads=None,
		global_target_factor = 0.15
	)
