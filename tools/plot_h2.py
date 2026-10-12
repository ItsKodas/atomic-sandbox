"""Plot the H2 pictures produced by `cargo run -p qm --release --example h2_picture`.

Reads out/h2_density_z0.csv, out/h2_density_z0_stretched.csv and out/h2_energy_curve.csv
(relative to the repo root) and writes docs/images/h2_density_z0.png and
docs/images/h2_energy_curve.png. Dev tool: needs numpy + matplotlib.

    python tools/plot_h2.py

(`python -I` hides user site-packages, which is where matplotlib lives on a Microsoft Store
Python, so don't use -I here.)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import PowerNorm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
IMG = ROOT / "docs" / "images"
BOHR_IN_ANGSTROM = 0.529177210903  # display only, matches crates/units
H_GRID = 0.25  # Bohr, grid spacing and nuclear smearing width used by the example
R_BONDED, R_STRETCHED = 1.4, 3.2

# Dark figure that also reads fine dropped onto a white page.
BG, PANEL, INK, MUTED, GRIDC = "#16191d", "#1e2228", "#e8eaed", "#a3a9b1", "#3a4049"
LINE = "#f4a259"  # single series: one warm accent, legible on the dark panel
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRIDC,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "font.size": 10, "axes.titlesize": 11,
})

to_ang = (lambda b: b * BOHR_IN_ANGSTROM, lambda a: a / BOHR_IN_ANGSTROM)


def load_plane(name):
    d = np.genfromtxt(OUT / name, delimiter=",", names=True)
    xs, ys = np.unique(d["x"]), np.unique(d["y"])
    rho = d["rho"].reshape(len(ys), len(xs))  # x fastest in the file -> rows are y
    return xs, ys, rho


def density_figure():
    planes = [(R_BONDED, "bonded", load_plane("h2_density_z0.csv")),
              (R_STRETCHED, "stretched", load_plane("h2_density_z0_stretched.csv"))]
    vmax = max(p[2][2].max() for p in planes)
    norm = PowerNorm(gamma=0.4, vmin=0.0, vmax=vmax)  # sqrt-ish so the bond region shows
    half = 4.0  # Bohr; crop the 12 Bohr box to where the density lives

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), constrained_layout=True)
    for ax, (r, label, (xs, ys, rho)) in zip(axes, planes):
        h = xs[1] - xs[0]
        im = ax.imshow(rho, origin="lower", cmap="magma", norm=norm, interpolation="bilinear",
                       extent=[xs[0] - h / 2, xs[-1] + h / 2, ys[0] - h / 2, ys[-1] + h / 2])
        levels = vmax * np.array([0.01, 0.03, 0.1, 0.3, 0.6])
        ax.contour(xs, ys, rho, levels=levels, colors="white", linewidths=0.5, alpha=0.35)
        ax.scatter([-r / 2, r / 2], [0, 0], s=40, facecolors="none", edgecolors="white", linewidths=1.4)
        mid = rho[len(ys) // 2, len(xs) // 2]
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_aspect("equal")
        ax.set_xlabel("x (Bohr)")
        ax.set_ylabel("y (Bohr)")
        sec = ax.secondary_xaxis("top", functions=to_ang)
        sec.set_xlabel("x (Å)", color=MUTED)
        sec.tick_params(colors=MUTED)
        ax.set_title(f"R = {r} Bohr ({r * BOHR_IN_ANGSTROM:.2f} Å), {label}", pad=6)
        ax.text(0.03, 0.03, f"ρ at bond midpoint = {mid:.3f} e/Bohr³", transform=ax.transAxes,
                color=INK, fontsize=9, bbox=dict(facecolor=BG, edgecolor="none", alpha=0.7, pad=3))
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("ρ(x, y, 0)  (electrons / Bohr³, power-law scale γ = 0.4)", color=INK)
    cb.ax.tick_params(colors=MUTED)
    fig.suptitle(f"H₂: Kohn-Sham LSDA electron density, real-space grid h = {H_GRID} Bohr (z = 0 plane)",
                 fontsize=12)
    path = IMG / "h2_density_z0.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def energy_figure():
    d = np.genfromtxt(OUT / "h2_energy_curve.csv", delimiter=",", names=True)
    r, de = d["R_bohr"], d["E_minus_Einf_eV"]
    i = int(np.argmin(de))
    r_inf = r[-1]

    fig, ax = plt.subplots(figsize=(8, 5.0), constrained_layout=True)
    ax.axhline(0, color=GRIDC, linewidth=1)
    ax.grid(True, color=GRIDC, linewidth=0.5, alpha=0.6)
    ax.plot(r, de, color=LINE, linewidth=2, marker="o", markersize=6,
            markeredgecolor=PANEL, markeredgewidth=1.2, zorder=3)
    ax.annotate(f"minimum on scan\nR = {r[i]:g} Bohr ({r[i] * BOHR_IN_ANGSTROM:.2f} Å)\n"
                f"E − E({r_inf:g}) = {de[i]:.2f} eV",
                xy=(r[i], de[i]), xytext=(r[i] + 0.3, de[i] + 5.2), color=INK, fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=1))
    ax.set_xlabel("R, H–H separation (Bohr)")
    ax.set_ylabel(f"E(R) − E(R = {r_inf:g} Bohr)  (eV)")
    sec = ax.secondary_xaxis("top", functions=to_ang)
    sec.set_xlabel("R (Å)", color=MUTED)
    sec.tick_params(colors=MUTED)
    ax.set_title("H₂ bond-forming energy curve: Kohn-Sham LSDA, spin-restricted, grid h = 0.25 Bohr", pad=8)
    ax.text(0.98, 0.96,
            "nuclei Gaussian-smeared σ = 0.25 Bohr; minimum shifted from 1.45 Bohr\n"
            "by the smearing (see validate/references.md). Reference E(5 Bohr) is not\n"
            "2·E(H): restricted KS does not dissociate cleanly.",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=MUTED)
    path = IMG / "h2_energy_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    for p in (density_figure(), energy_figure()):
        print(f"wrote {p.relative_to(ROOT)}")
