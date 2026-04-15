#!/usr/bin/env python3
"""
Eastern Mau Forest LULC Desktop Viewer & Analyzer
==================================================
A standalone Python desktop GIS application for viewing, analyzing, and
comparing multi-temporal classified land use/land cover (LULC) raster data
from the Eastern Mau Forest Reserve, Kenya.

Features:
  - Load and display classified GeoTIFF rasters with original symbology
  - Compute area statistics per LULC class for any epoch
  - Compare two epochs and quantify land cover change
  - Export analysis reports as CSV
  - Interactive matplotlib map rendering within a tkinter GUI

Author: Mike Papayai Passiany
Date: April 2026
Dependencies: rasterio, numpy, matplotlib, tkinter (built-in)
"""

import os
import sys
import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from collections import OrderedDict

import numpy as np

try:
    import rasterio
    from rasterio.transform import array_bounds
except ImportError:
    sys.exit("ERROR: rasterio is required. Install via: pip install rasterio")

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
except ImportError:
    sys.exit("ERROR: matplotlib is required. Install via: pip install matplotlib")


# ── LULC Classification Scheme (from .clr files & published research) ────────
CLASS_INFO = OrderedDict([
    (1, {"label": "Dense Forest",     "rgb": (0, 109, 44),   "hex": "#006d2c"}),
    (2, {"label": "Barren Land",      "rgb": (247, 248, 233), "hex": "#f7f8e9"}),
    (3, {"label": "Settlement",       "rgb": (186, 228, 179), "hex": "#bae4b3"}),
    (4, {"label": "Grassland",        "rgb": (49, 163, 84),  "hex": "#31a354"}),
    (5, {"label": "Planted Farmland", "rgb": (116, 196, 118), "hex": "#74c476"}),
])

NODATA_VAL = 0
STUDY_AREA_HA = 66411  # Published study area


class LULCDesktopViewer(tk.Tk):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.title("Eastern Mau Forest — LULC Desktop Viewer & Analyzer")
        self.geometry("1200x800")
        self.configure(bg="#1a1a2e")
        self.minsize(900, 600)

        self.raster_paths = {}   # {year: filepath}
        self.raster_data = {}    # {year: (data_array, transform, crs, width, height, res)}
        self.current_year = None

        self._build_ui()

    # ── UI Construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        # Top toolbar
        toolbar = ttk.Frame(self)
        toolbar.pack(fill=tk.X, padx=8, pady=4)

        ttk.Button(toolbar, text="Load GeoTIFFs…", command=self._load_files).pack(side=tk.LEFT, padx=4)
        ttk.Label(toolbar, text="Epoch:").pack(side=tk.LEFT, padx=(16, 4))
        self.epoch_var = tk.StringVar()
        self.epoch_combo = ttk.Combobox(toolbar, textvariable=self.epoch_var,
                                        state="readonly", width=10)
        self.epoch_combo.pack(side=tk.LEFT, padx=4)
        self.epoch_combo.bind("<<ComboboxSelected>>", self._on_epoch_change)

        ttk.Button(toolbar, text="Area Statistics", command=self._show_stats).pack(side=tk.LEFT, padx=16)
        ttk.Button(toolbar, text="Change Detection", command=self._show_change).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Export Report", command=self._export_report).pack(side=tk.LEFT, padx=4)

        # Main content: map + sidebar
        content = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        content.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Map panel
        map_frame = ttk.Frame(content)
        content.add(map_frame, weight=3)

        self.fig = Figure(figsize=(8, 6), facecolor="#1a1a2e")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#0f3460")
        self.ax.set_title("Load GeoTIFF files to begin", color="#94a3b8", fontsize=12)
        self.ax.tick_params(colors="#64748b", labelsize=8)
        self.canvas = FigureCanvasTkAgg(self.fig, master=map_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        nav_toolbar = NavigationToolbar2Tk(self.canvas, map_frame)
        nav_toolbar.update()

        # Sidebar — statistics table
        side_frame = ttk.Frame(content, width=300)
        content.add(side_frame, weight=1)

        ttk.Label(side_frame, text="CLASS STATISTICS", font=("Segoe UI", 11, "bold")).pack(pady=8)

        cols = ("Class", "Pixels", "Area (ha)", "% Cover")
        self.stats_tree = ttk.Treeview(side_frame, columns=cols, show="headings", height=8)
        for c in cols:
            self.stats_tree.heading(c, text=c)
            self.stats_tree.column(c, width=80, anchor=tk.CENTER)
        self.stats_tree.column("Class", width=130, anchor=tk.W)
        self.stats_tree.pack(fill=tk.BOTH, expand=True, padx=4)

        # Status bar
        self.status_var = tk.StringVar(value="Ready — load classified GeoTIFF rasters to begin analysis")
        ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN,
                  anchor=tk.W).pack(fill=tk.X, side=tk.BOTTOM, padx=8, pady=2)

    # ── File Loading ─────────────────────────────────────────────────────────

    def _load_files(self):
        paths = filedialog.askopenfilenames(
            title="Select Classified GeoTIFF Files",
            filetypes=[("GeoTIFF", "*.tif *.tiff"), ("All files", "*.*")]
        )
        if not paths:
            return

        for p in paths:
            year = self._extract_year(p)
            if year:
                self.raster_paths[year] = p
                self._read_raster(year, p)

        if self.raster_paths:
            years = sorted(self.raster_paths.keys())
            self.epoch_combo["values"] = years
            self.epoch_combo.set(years[-1])
            self._display(years[-1])
            self.status_var.set(f"Loaded {len(years)} epochs: {', '.join(map(str, years))}")

    def _extract_year(self, path):
        """Extract 4-digit year from filename."""
        name = os.path.basename(path)
        for token in name.replace("-", " ").replace("_", " ").split():
            digits = "".join(c for c in token if c.isdigit())
            if len(digits) == 4 and 1980 <= int(digits) <= 2030:
                return int(digits)
            if len(digits) == 5 and 1980 <= int(digits[:4]) <= 2030:
                return int(digits[:4])
        return None

    def _read_raster(self, year, path):
        """Read classified raster into memory."""
        with rasterio.open(path) as src:
            data = src.read(1)
            self.raster_data[year] = {
                "data": data,
                "transform": src.transform,
                "crs": src.crs,
                "width": src.width,
                "height": src.height,
                "res": src.res,
                "bounds": src.bounds,
                "nodata": src.nodata,
            }

    # ── Map Display ──────────────────────────────────────────────────────────

    def _display(self, year):
        """Render classified raster on the matplotlib axes."""
        self.current_year = year
        info = self.raster_data[year]
        data = info["data"].astype(float)
        data[data == NODATA_VAL] = np.nan
        if info["nodata"] is not None:
            data[data == info["nodata"]] = np.nan

        self.ax.clear()

        # Build colormap from class scheme
        colors_list = ["#000000"]  # 0 = nodata placeholder
        for v in range(1, 6):
            colors_list.append(CLASS_INFO[v]["hex"])
        cmap = ListedColormap(colors_list)
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        norm = BoundaryNorm(bounds, cmap.N)

        ext = [info["bounds"].left, info["bounds"].right,
               info["bounds"].bottom, info["bounds"].top]
        self.ax.imshow(data, cmap=cmap, norm=norm, extent=ext,
                       interpolation="nearest", origin="upper")
        self.ax.set_title(f"LULC Classification — {year}", color="#e0e0e0",
                          fontsize=13, fontweight="bold")
        self.ax.set_xlabel("Easting (m)", color="#94a3b8", fontsize=9)
        self.ax.set_ylabel("Northing (m)", color="#94a3b8", fontsize=9)
        self.ax.tick_params(colors="#64748b", labelsize=7)

        # Legend
        legend_patches = [Patch(facecolor=CLASS_INFO[v]["hex"],
                                label=CLASS_INFO[v]["label"]) for v in range(1, 6)]
        self.ax.legend(handles=legend_patches, loc="lower right", fontsize=8,
                       facecolor="#16213e", edgecolor="#334155",
                       labelcolor="#cbd5e1", framealpha=0.9)

        self.fig.tight_layout()
        self.canvas.draw()

        # Update statistics sidebar
        self._update_stats_table(year)

    def _on_epoch_change(self, event=None):
        year = int(self.epoch_var.get())
        self._display(year)

    # ── Statistics ───────────────────────────────────────────────────────────

    def _compute_stats(self, year):
        """Compute area statistics for each LULC class."""
        info = self.raster_data[year]
        data = info["data"]
        px_area_m2 = info["res"][0] * info["res"][1]
        px_area_ha = px_area_m2 / 10000.0

        stats = []
        total_valid = 0
        for v in range(1, 6):
            count = int(np.sum(data == v))
            total_valid += count

        for v in range(1, 6):
            count = int(np.sum(data == v))
            area_ha = count * px_area_ha
            pct = (count / total_valid * 100) if total_valid > 0 else 0
            stats.append({
                "class_id": v,
                "label": CLASS_INFO[v]["label"],
                "pixels": count,
                "area_ha": round(area_ha, 2),
                "pct": round(pct, 2),
            })
        return stats

    def _update_stats_table(self, year):
        """Update the sidebar statistics treeview."""
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)
        stats = self._compute_stats(year)
        for s in stats:
            self.stats_tree.insert("", tk.END, values=(
                s["label"], f"{s['pixels']:,}", f"{s['area_ha']:,.1f}", f"{s['pct']:.2f}%"
            ))

    def _show_stats(self):
        """Show detailed statistics in a popup window."""
        if not self.current_year:
            messagebox.showinfo("Info", "Load GeoTIFF files first.")
            return
        stats = self._compute_stats(self.current_year)
        msg = f"LULC Area Statistics — {self.current_year}\n"
        msg += f"CRS: {self.raster_data[self.current_year]['crs']}\n"
        msg += f"Resolution: {self.raster_data[self.current_year]['res'][0]}m\n"
        msg += "-" * 50 + "\n"
        for s in stats:
            msg += f"{s['label']:20s} {s['area_ha']:>10,.1f} ha  ({s['pct']:5.2f}%)\n"
        messagebox.showinfo(f"Statistics — {self.current_year}", msg)

    # ── Change Detection ─────────────────────────────────────────────────────

    def _show_change(self):
        """Show change detection dialog between two epochs."""
        years = sorted(self.raster_data.keys())
        if len(years) < 2:
            messagebox.showinfo("Info", "Load at least 2 GeoTIFF epochs for change detection.")
            return

        dlg = tk.Toplevel(self)
        dlg.title("Change Detection Analysis")
        dlg.geometry("550x450")
        dlg.configure(bg="#1a1a2e")

        ttk.Label(dlg, text="Select two epochs to compare:").pack(pady=8)
        frame = ttk.Frame(dlg)
        frame.pack()

        ttk.Label(frame, text="From:").grid(row=0, column=0, padx=4)
        from_var = tk.StringVar(value=str(years[0]))
        from_cb = ttk.Combobox(frame, textvariable=from_var,
                               values=[str(y) for y in years], state="readonly", width=8)
        from_cb.grid(row=0, column=1, padx=4)

        ttk.Label(frame, text="To:").grid(row=0, column=2, padx=4)
        to_var = tk.StringVar(value=str(years[-1]))
        to_cb = ttk.Combobox(frame, textvariable=to_var,
                             values=[str(y) for y in years], state="readonly", width=8)
        to_cb.grid(row=0, column=3, padx=4)

        result_text = tk.Text(dlg, height=18, width=60, bg="#0f3460", fg="#e0e0e0",
                              font=("Consolas", 10), relief=tk.FLAT)
        result_text.pack(padx=16, pady=12, fill=tk.BOTH, expand=True)

        def run_analysis():
            y1, y2 = int(from_var.get()), int(to_var.get())
            if y1 == y2:
                result_text.delete("1.0", tk.END)
                result_text.insert(tk.END, "Select two different epochs.")
                return
            s1, s2 = self._compute_stats(y1), self._compute_stats(y2)
            result_text.delete("1.0", tk.END)
            header = f"LULC Change Detection: {y1} → {y2}\n{'=' * 50}\n"
            header += f"{'Class':20s} {'From %':>8s} {'To %':>8s} {'Change':>8s}\n"
            header += "-" * 50 + "\n"
            result_text.insert(tk.END, header)
            for a, b in zip(s1, s2):
                delta = b["pct"] - a["pct"]
                sign = "+" if delta > 0 else ""
                result_text.insert(tk.END,
                    f"{a['label']:20s} {a['pct']:>7.2f}% {b['pct']:>7.2f}% {sign}{delta:>6.2f}%\n")

        ttk.Button(dlg, text="Analyze", command=run_analysis).pack(pady=4)
        run_analysis()

    # ── Export Report ─────────────────────────────────────────────────────────

    def _export_report(self):
        """Export current epoch stats + all-epoch comparison as CSV."""
        if not self.raster_data:
            messagebox.showinfo("Info", "Load GeoTIFF files first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")],
            initialfile=f"lulc_report_{self.current_year}.csv"
        )
        if not path:
            return

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Eastern Mau Forest Reserve — LULC Analysis Report"])
            writer.writerow(["Generated by LULC Desktop Viewer — Mike Papayai Passiany"])
            writer.writerow([])

            # All-epoch comparison
            years = sorted(self.raster_data.keys())
            header = ["Class"] + [f"{y} (%)" for y in years]
            if len(years) >= 2:
                header.append(f"Change {years[0]}-{years[-1]}")
            writer.writerow(header)

            all_stats = {y: self._compute_stats(y) for y in years}
            for i in range(5):
                row = [CLASS_INFO[i + 1]["label"]]
                for y in years:
                    row.append(f"{all_stats[y][i]['pct']:.2f}")
                if len(years) >= 2:
                    delta = all_stats[years[-1]][i]["pct"] - all_stats[years[0]][i]["pct"]
                    row.append(f"{'+' if delta > 0 else ''}{delta:.2f}")
                writer.writerow(row)

        self.status_var.set(f"Report exported: {path}")
        messagebox.showinfo("Export", f"Report saved to:\n{path}")


if __name__ == "__main__":
    app = LULCDesktopViewer()
    app.mainloop()
