"""
Results export utilities for SimResult objects.
Supports: CSV, HDF5 (with npz fallback), PNG figure, text report.
"""
import csv
import datetime
import numpy as np


class ResultsExporter:
    """Export SimResult to various formats."""

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    def export_csv(self, result, path: str) -> None:
        """
        Export aerial_image as CSV and append metrics rows.
        Two sections separated by a blank line:
          1. METRICS table (key, value)
          2. AERIAL_IMAGE as a numeric matrix
        """
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Metrics section
            writer.writerow(['# METRICS'])
            writer.writerow(['Metric', 'Value'])
            metrics_rows = self._metrics_rows(result)
            for k, v in metrics_rows:
                writer.writerow([k, v])

            writer.writerow([])  # blank separator

            # Aerial image section
            if result.aerial_image is not None:
                writer.writerow(['# AERIAL_IMAGE (rows x cols)'])
                for row in result.aerial_image:
                    writer.writerow(['{:.6f}'.format(v) for v in row])

    # ------------------------------------------------------------------
    # HDF5 / npz fallback
    # ------------------------------------------------------------------

    def export_hdf5(self, result, path: str) -> None:
        """
        Export datasets to HDF5 if h5py is available, else .npz fallback.
        HDF5 groups: /aerial_image, /mask_grid, /metrics (attrs), /config (attrs).
        """
        try:
            import h5py
            self._write_hdf5(result, path, h5py)
        except ImportError:
            # Fallback: save as npz (override path extension for clarity)
            npz_path = path if path.endswith('.npz') else path + '.npz'
            arrays = {}
            if result.aerial_image is not None:
                arrays['aerial_image'] = result.aerial_image
            if result.mask_grid is not None:
                arrays['mask_grid'] = result.mask_grid
            if result.source_points is not None:
                arrays['source_points'] = result.source_points
            np.savez_compressed(npz_path, **arrays)

    def _write_hdf5(self, result, path: str, h5py) -> None:
        with h5py.File(path, 'w') as f:
            if result.aerial_image is not None:
                f.create_dataset('aerial_image', data=result.aerial_image,
                                 compression='gzip')
            if result.mask_grid is not None:
                f.create_dataset('mask_grid', data=result.mask_grid,
                                 compression='gzip')
            if result.source_points is not None:
                f.create_dataset('source_points', data=result.source_points,
                                 compression='gzip')

            # Metrics as dataset attributes
            metrics_grp = f.create_group('metrics')
            for k, v in self._metrics_rows(result):
                try:
                    metrics_grp.attrs[k] = float(v)
                except (ValueError, TypeError):
                    metrics_grp.attrs[k] = str(v)

            # Config as flat string attributes
            config_grp = f.create_group('config')
            self._flatten_config(result.config, config_grp)

    def _flatten_config(self, cfg, grp, prefix=''):
        """Recursively write config dict as HDF5 group attributes."""
        for k, v in cfg.items():
            key = '{}.{}'.format(prefix, k) if prefix else k
            if isinstance(v, dict):
                self._flatten_config(v, grp, key)
            else:
                try:
                    grp.attrs[key] = v
                except TypeError:
                    grp.attrs[key] = str(v)

    # ------------------------------------------------------------------
    # PNG figure
    # ------------------------------------------------------------------

    def export_png(self, result, path: str, dpi: int = 150) -> None:
        """
        Save a 5-panel figure (aerial image, mask, overlay, cross-section,
        metrics text) as PNG.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(14, 9), dpi=dpi)
        fig.patch.set_facecolor('#1a1a2e')
        gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

        ax_aerial  = fig.add_subplot(gs[0, 0])
        ax_mask    = fig.add_subplot(gs[0, 1])
        ax_overlay = fig.add_subplot(gs[0, 2])
        ax_cs      = fig.add_subplot(gs[1, 0:2])
        ax_text    = fig.add_subplot(gs[1, 2])

        extent = self._extent(result)

        # Aerial image
        if result.aerial_image is not None:
            im = ax_aerial.imshow(result.aerial_image, cmap='inferno',
                                  origin='lower', vmin=0, vmax=1, extent=extent,
                                  aspect='auto')
            fig.colorbar(im, ax=ax_aerial, fraction=0.046)
        ax_aerial.set_title('Aerial Image', color='white')
        ax_aerial.set_facecolor('#16213e')

        # Mask
        if result.mask_grid is not None:
            ax_mask.imshow(result.mask_grid, cmap='gray', origin='lower',
                           extent=extent, aspect='auto')
        ax_mask.set_title('Mask', color='white')
        ax_mask.set_facecolor('#16213e')

        # Overlay
        if result.aerial_image is not None and result.mask_grid is not None:
            ax_overlay.imshow(result.aerial_image, cmap='inferno', origin='lower',
                              alpha=0.7, extent=extent, aspect='auto')
            ny, nx = result.mask_grid.shape
            cx = np.linspace(extent[0], extent[1], nx) if extent else np.arange(nx)
            cy = np.linspace(extent[2], extent[3], ny) if extent else np.arange(ny)
            ax_overlay.contour(cx, cy, result.mask_grid, levels=[0.5],
                               colors='cyan', linewidths=0.8)
        ax_overlay.set_title('Overlay', color='white')
        ax_overlay.set_facecolor('#16213e')

        # Cross-section (center row)
        if result.aerial_image is not None:
            mid = result.aerial_image.shape[0] // 2
            profile = result.aerial_image[mid, :]
            x = np.linspace(extent[0], extent[1], len(profile)) if extent else np.arange(len(profile))
            ax_cs.plot(x, profile, color='#1a6bb5', linewidth=1.4)
            threshold = result.metrics.get('threshold', 0.3)
            ax_cs.axhline(threshold, color='red', linestyle='--', linewidth=0.9,
                          label='Threshold {:.2f}'.format(threshold))
            ax_cs.set_ylim(0, 1)
            ax_cs.legend(fontsize=8)
            ax_cs.set_facecolor('#16213e')
        ax_cs.set_title('Cross-section (center row)', color='white')

        # Metrics text panel
        ax_text.set_facecolor('#16213e')
        ax_text.set_axis_off()
        lines = ['{}: {}'.format(k, v) for k, v in self._metrics_rows(result)]
        ax_text.text(0.05, 0.95, '\n'.join(lines),
                     transform=ax_text.transAxes, va='top', ha='left',
                     fontsize=8, color='white', family='monospace')
        ax_text.set_title('Metrics', color='white')

        for ax in [ax_aerial, ax_mask, ax_overlay, ax_cs, ax_text]:
            ax.tick_params(colors='#aaaaaa', labelsize=7)
            for spine in ax.spines.values():
                spine.set_color('#333355')

        fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)

    # ------------------------------------------------------------------
    # Text report
    # ------------------------------------------------------------------

    def export_report(self, result, path: str) -> None:
        """
        Write a human-readable text report with metrics, config summary,
        and timestamp.
        """
        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        lines = [
            '=' * 60,
            'Lithography Simulation Report',
            'Generated: {}'.format(ts),
            '=' * 60,
            '',
            '--- METRICS ---',
        ]
        for k, v in self._metrics_rows(result):
            lines.append('  {:<20} {}'.format(k, v))

        lines += ['', '--- CONFIG ---']
        self._format_config(result.config, lines, indent=2)

        if result.layout_path:
            lines += ['', 'Layout file: {}'.format(result.layout_path)]

        lines += ['', 'Status: {}'.format(result.status)]
        if result.error_msg:
            lines.append('Error: {}'.format(result.error_msg))
        lines.append('=' * 60)

        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _metrics_rows(self, result):
        rows = [
            ('CD_nm',    '{:.2f}'.format(result.cd_nm)),
            ('NILS',     '{:.4f}'.format(result.nils)),
            ('Contrast', '{:.4f}'.format(result.contrast)),
            ('DOF_nm',   '{:.1f}'.format(result.metrics.get('dof_nm', 0.0))),
            ('I_max',    '{:.4f}'.format(result.metrics.get('i_max', 0.0))),
            ('I_min',    '{:.4f}'.format(result.metrics.get('i_min', 0.0))),
            ('Status',   result.status),
        ]
        for k, v in result.metrics.items():
            if k not in {'dof_nm', 'i_max', 'i_min', 'threshold'}:
                rows.append((k, str(v)))
        return rows

    def _extent(self, result):
        """Return [x0, x1, y0, y1] extent from config, or None."""
        try:
            domain = result.config['simulation']['domain_size_nm']
            half = domain / 2.0
            return [-half, half, -half, half]
        except Exception:
            return None

    def _format_config(self, cfg, lines, indent=0):
        prefix = ' ' * indent
        for k, v in cfg.items():
            if isinstance(v, dict):
                lines.append('{}{}:'.format(prefix, k))
                self._format_config(v, lines, indent + 2)
            else:
                lines.append('{}{}: {}'.format(prefix, k, v))
