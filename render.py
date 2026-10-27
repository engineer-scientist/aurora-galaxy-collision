#!/usr/bin/env python3
"""Turn snapshot files into PNG frames.

Each frame is a column-density projection. The two galaxies' disks get
different hues so you can see which material ends up where after the
encounter, and the dark-matter haloes are drawn faint underneath. Counts go
through an asinh stretch, which behaves like a linear scale near zero and a
log scale at high density - the standard trick for showing faint tidal
features and bright nuclei in the same image.
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TIME_UNIT_MYR = 4.715
WARM = np.array([1.00, 0.70, 0.40])   # galaxy A disk
COOL = np.array([0.45, 0.70, 1.00])   # galaxy B disk
HALO = np.array([0.55, 0.50, 0.80])
HALO_WEIGHT = 0.16


def load_kinds(ic_path):
    with open(ic_path, "rb") as f:
        n = int(np.fromfile(f, np.int64, 1)[0])
        f.seek(8 + 28 * n)                      # pos+vel (24N) + mass (4N)
        kind = np.fromfile(f, np.int32, n)
    return n, kind


def smooth(h):
    """Separable 1-2-1 blur - a cheap stand-in for a telescope's point-spread
    function. Without it, a 1080-pixel grid puts so few particles in each cell
    that the tidal features read as speckle rather than structure."""
    p = np.pad(h, 1, mode="constant")
    h = 0.25 * p[:-2, 1:-1] + 0.5 * p[1:-1, 1:-1] + 0.25 * p[2:, 1:-1]
    p = np.pad(h, 1, mode="constant")
    return 0.25 * p[1:-1, :-2] + 0.5 * p[1:-1, 1:-1] + 0.25 * p[1:-1, 2:]


def stretch(h, knee, hmax):
    """asinh stretch normalised so that hmax maps to 1.

    `knee` sets where the curve turns over from linear to logarithmic: small
    knee lifts faint tidal material, large knee keeps only the bright cores.
    Normalising by arcsinh(hmax/knee) is what keeps the output in [0, 1] -
    dividing by arcsinh(1/knee) instead blows every pixel past white.
    """
    return np.clip(np.arcsinh(h / knee) / np.arcsinh(max(hmax, 2.0 * knee) / knee),
                   0.0, 1.0)


def render(xyz, masks, px, half, sc):
    rng = [[-half, half], [-half, half]]
    rgb = np.zeros((px, px, 3), dtype=np.float32)

    def hist(sel):
        h, _, _ = np.histogram2d(xyz[sel, 1], xyz[sel, 0], bins=px, range=rng)
        return smooth(h)

    # The halo is a faint backdrop: it holds most of the mass but almost none
    # of the light, and at full weight it simply drowns the disks.
    rgb += (HALO_WEIGHT * stretch(hist(masks["halo"]), sc["knee_halo"],
                                  sc["hmax_halo"]))[..., None] * HALO
    rgb += stretch(hist(masks["diskA"]), sc["knee_disk"], sc["hmax_disk"])[..., None] * WARM
    rgb += stretch(hist(masks["diskB"]), sc["knee_disk"], sc["hmax_disk"])[..., None] * COOL
    return np.clip(rgb, 0.0, 1.0)


def calibrate(path, masks, px, half):
    """Fix the stretch from the first frame so brightness does not pump."""
    xyz = np.fromfile(path, np.float32).reshape(-1, 3)
    rng = [[-half, half], [-half, half]]

    def pct(sel, q):
        h, _, _ = np.histogram2d(xyz[sel, 1], xyz[sel, 0], bins=px, range=rng)
        h = smooth(h)
        nz = h[h > 0.02]
        return float(np.percentile(nz, q)) if nz.size else 1.0

    hmax_disk = max(4.0, pct(masks["diskA"], 99.9))
    hmax_halo = max(4.0, pct(masks["halo"], 99.9))
    return {
        "hmax_disk": hmax_disk, "knee_disk": max(0.9, hmax_disk / 110.0),
        "hmax_halo": hmax_halo, "knee_halo": max(0.9, hmax_halo / 45.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snaps", default="snaps")
    ap.add_argument("--ic", default="ic.bin")
    ap.add_argument("--out", default="frames")
    ap.add_argument("--px", type=int, default=1080)
    ap.add_argument("--half", type=float, default=90.0, help="half-width, kpc")
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--snap-every", type=int, default=100)
    args = ap.parse_args()

    n, kind = load_kinds(args.ic)
    masks = {
        "halo": (kind == 0) | (kind == 2),
        "diskA": kind == 1,
        "diskB": kind == 3,
    }
    os.makedirs(args.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.snaps, "snap_*.bin")))
    if not files:
        raise SystemExit(f"no snapshots in {args.snaps}")

    sc = calibrate(files[0], masks, args.px, args.half)
    print(f"{len(files)} snapshots, N={n}")
    print(f"  disk: hmax={sc['hmax_disk']:.1f} knee={sc['knee_disk']:.2f}   "
          f"halo: hmax={sc['hmax_halo']:.1f} knee={sc['knee_halo']:.2f}")

    for i, path in enumerate(files):
        xyz = np.fromfile(path, np.float32).reshape(-1, 3)
        rgb = render(xyz, masks, args.px, args.half, sc)

        fig = plt.figure(figsize=(args.px / 100, args.px / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(rgb, origin="lower", extent=[-args.half, args.half] * 2,
                  interpolation="nearest")
        ax.set_facecolor("black")
        ax.set_xticks([]); ax.set_yticks([])

        myr = i * args.snap_every * args.dt * TIME_UNIT_MYR
        ax.text(0.03, 0.96, f"t = {myr:7.1f} Myr", color="white",
                transform=ax.transAxes, va="top", family="monospace", fontsize=14)
        ax.text(0.03, 0.04, f"{n:,} particles   direct N-body   Aurora",
                color="0.6", transform=ax.transAxes, family="monospace", fontsize=10)
        # scale bar, bottom right so it clears the caption
        x0 = args.half * 0.52
        ax.plot([x0, x0 + 20], [-args.half * 0.86] * 2, color="white", lw=2)
        ax.text(x0 + 10, -args.half * 0.82, "20 kpc", color="white",
                ha="center", family="monospace", fontsize=10)

        fig.savefig(os.path.join(args.out, f"frame_{i:05d}.png"),
                    facecolor="black")
        plt.close(fig)
        if i % 25 == 0:
            print(f"  frame {i}/{len(files)}", flush=True)

    print(f"wrote {len(files)} frames to {args.out}")


if __name__ == "__main__":
    main()
