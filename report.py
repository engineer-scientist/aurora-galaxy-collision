#!/usr/bin/env python3
"""Build the PDF report from the finished run.

Aimed at an undergraduate who has had introductory mechanics: it assumes
Newton's law of gravitation and not much else.
"""
import argparse
import glob
import os
import re
import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

TIME_UNIT_MYR = 4.715
VEL_UNIT_KMS = 207.4

# Images come from render.py so the report and the animation cannot drift
# apart in colour or stretch.
from render import calibrate, load_kinds, render as frame_rgb


def text_page(pdf, title, blocks):
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.08, 0.93, title, fontsize=19, weight="bold", va="top")
    y = 0.87
    for kind, body in blocks:
        if kind == "h":
            y -= 0.018
            fig.text(0.08, y, body, fontsize=12.5, weight="bold", va="top")
            y -= 0.030
        elif kind == "p":
            wrapped = textwrap.fill(body, 92)
            fig.text(0.08, y, wrapped, fontsize=10, va="top", linespacing=1.55)
            y -= 0.026 + 0.0205 * wrapped.count("\n")
        elif kind == "eq":
            fig.text(0.5, y - 0.012, body, fontsize=15, va="top", ha="center")
            y -= 0.070
        elif kind == "mono":
            fig.text(0.10, y, body, fontsize=8.6, va="top", family="monospace",
                     linespacing=1.5)
            y -= 0.022 + 0.0165 * body.count("\n")
    pdf.savefig(fig); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snaps", default="snaps")
    ap.add_argument("--ic", default="ic.bin")
    ap.add_argument("--simlog", default="")
    ap.add_argument("--out", default="galaxy_collision_report.pdf")
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--snap-every", type=int, default=100)
    ap.add_argument("--half", type=float, default=90.0)
    args = ap.parse_args()

    n, kind = load_kinds(args.ic)
    masks = {"halo": (kind == 0) | (kind == 2), "diskA": kind == 1, "diskB": kind == 3}
    files = sorted(glob.glob(os.path.join(args.snaps, "snap_*.bin")))
    if not files:
        raise SystemExit("no snapshots")

    times = np.arange(len(files)) * args.snap_every * args.dt * TIME_UNIT_MYR / 1000.0  # Gyr

    # Separation between the two stellar disks, and how bound each stays.
    sep = np.empty(len(files))
    for i, p in enumerate(files):
        xyz = np.fromfile(p, np.float32).reshape(-1, 3)
        a = np.median(xyz[masks["diskA"]], axis=0)
        b = np.median(xyz[masks["diskB"]], axis=0)
        sep[i] = np.linalg.norm(a - b)

    perf = ""
    if args.simlog and os.path.exists(args.simlog):
        txt = open(args.simlog, errors="ignore").read()
        m = re.search(r"done:.*", txt)
        if m:
            perf = m.group(0)

    # Counts scale with pixel area, so each render size needs its own stretch.
    sc_hero = calibrate(files[0], masks, 900, args.half)
    sc_tile = calibrate(files[0], masks, 500, args.half)

    with PdfPages(args.out) as pdf:
        # ---- cover ----
        # First pericentre, not the global minimum: after the galaxies merge
        # the separation sits near zero for the rest of the run, so argmin
        # would point at the merged remnant instead of the first close pass.
        peak = int(np.argmin(sep))
        for i in range(2, len(sep) - 2):
            if sep[i] < sep[i - 1] and sep[i] <= sep[i + 1] and sep[i] < 0.6 * sep[0]:
                peak = i
                break
        hero = frame_rgb(np.fromfile(files[min(peak + 2, len(files) - 1)],
                                     np.float32).reshape(-1, 3),
                         masks, 900, args.half, sc_hero)
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor("white")
        fig.text(0.5, 0.955, "When Two Galaxies Collide",
                 fontsize=25, weight="bold", ha="center")
        fig.text(0.5, 0.925,
                 "A direct N-body simulation on the Aurora exascale supercomputer",
                 fontsize=12, ha="center", color="0.3")
        ax = fig.add_axes([0.08, 0.34, 0.84, 0.55])
        ax.imshow(hero, origin="lower", extent=[-args.half, args.half] * 2)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("0.7")
        fig.text(0.5, 0.305,
                 f"Just after first pericentre, t = {times[peak]:.2f} Gyr",
                 fontsize=9.5, ha="center", color="0.4", style="italic")
        body = (
            f"This report follows {n:,} gravitating particles as two disk galaxies "
            f"fall together, swing past each other, and begin to merge. Nothing in "
            f"the simulation knows what a spiral arm or a tidal tail is. The only "
            f"rule is Newton's law of gravitation, applied between every pair of "
            f"particles at every step. The structures you see emerge from that rule "
            f"alone."
        )
        fig.text(0.08, 0.24, textwrap.fill(body, 95), fontsize=10.5, va="top",
                 linespacing=1.6)
        fig.text(0.08, 0.08, "Argonne Leadership Computing Facility  |  Aurora\n"
                             "Allocation: DD_plasma_physics",
                 fontsize=8.5, color="0.45", va="bottom", family="monospace")
        pdf.savefig(fig); plt.close(fig)

        # ---- the physics ----
        text_page(pdf, "1.  What the computer is actually doing", [
            ("h", "One rule, applied a great many times"),
            ("p", "Every particle feels a pull from every other particle. For particle i, "
                  "the acceleration is the sum of the pulls from all the others:"),
            ("eq", r"$\mathbf{a}_i \;=\; G \sum_j m_j \, "
                   r"\frac{\mathbf{r}_j - \mathbf{r}_i}"
                   r"{|\mathbf{r}_j - \mathbf{r}_i|^{3}}$"),
            ("p", "That is the whole physical model. There is no fluid, no magnetic field, "
                  "no chemistry - just gravity. The difficulty is purely one of counting: "
                  f"with N = {n:,} particles, each step requires N² ≈ "
                  f"{n*n/1e12:.2g} trillion pair interactions, and the simulation takes "
                  "tens of thousands of steps."),
            ("h", "Why the particles are not stars"),
            ("p", "A real galaxy holds something like 10¹¹ stars, far more than we "
                  "can track. Each particle here stands in for roughly a million stars' "
                  "worth of mass. This is fine for studying the shape of a galaxy, because "
                  "the large-scale gravitational field depends on where the mass is, not on "
                  "how finely it is chopped up."),
            ("h", "Softening: blurring the point masses"),
            ("p", "If two particles pass very close, the 1/r² force diverges and the "
                  "simulation blows up. Since our particles are stand-ins for large clouds "
                  "of stars rather than true points, we blur each one slightly, replacing "
                  "r² with r² + ε²:"),
            ("eq", r"$\mathbf{a}_i \;=\; G \sum_j m_j \, "
                   r"\frac{\mathbf{r}_j - \mathbf{r}_i}"
                   r"{\left(|\mathbf{r}_j - \mathbf{r}_i|^{2} "
                   r"+ \varepsilon^{2}\right)^{3/2}}$"),
            ("p", "Here ε = 0.08 kpc, about 260 light years. Below that scale the "
                  "simulation deliberately does not resolve structure. A pleasant side "
                  "effect: when j = i the numerator is zero, so a particle exerts no force "
                  "on itself and no special case is needed in the code."),
            ("h", "Stepping forward in time"),
            ("p", "We use the leapfrog (kick-drift-kick) scheme: nudge the velocities by "
                  "half a step, move the positions a full step, recompute the forces, then "
                  "nudge the velocities again. Leapfrog is symplectic - it respects the "
                  "geometric structure of mechanics - so the total energy wanders slightly "
                  "but never drifts steadily away, even over tens of thousands of steps. A "
                  "naive Euler scheme would slowly inflate the galaxies into nonsense."),
        ])

        # ---- setup ----
        text_page(pdf, "2.  How the two galaxies were built", [
            ("h", "Each galaxy: a halo and a disk"),
            ("p", "Every galaxy is assembled from two pieces. A Plummer sphere of "
                  "4×10¹⁰ solar masses with a 20 kpc scale radius plays the "
                  "part of the dark matter halo and central bulge; it is sampled exactly "
                  "from its own distribution function, so it would sit still forever if "
                  "left alone. On top sits an exponential stellar disk of 5×10¹⁰ "
                  "solar masses with a 3.5 kpc scale length and 0.3 kpc thickness, with "
                  "stars placed on near-circular orbits set by the mass enclosed inside "
                  "them."),
            ("p", "The disk is the part you can see. The halo is invisible in real "
                  "observations and nearly invisible in the images here, but it carries "
                  "most of the mass and therefore controls the encounter."),
            ("h", "The orbit"),
            ("p", "The galaxies start 60 kpc apart on a parabolic orbit - exactly the "
                  "marginally-bound case, the textbook idealisation of two galaxies falling "
                  "together from rest at infinity. Their transverse velocity is set so "
                  "that two point masses of the same total mass would pass within "
                  "12 kpc. Real galaxies are not point masses, though - their mass is "
                  "spread over tens of kiloparsecs - so the true closest approach comes "
                  "out larger than that estimate. Section 4 shows what actually "
                  "happened. One disk is tilted 20° "
                  "to the orbital plane and the other 70°, so the encounter is close to "
                  "prograde for the first galaxy and strongly inclined for the second. That "
                  "asymmetry is deliberate: prograde encounters, where the stars orbit in "
                  "the same sense as the companion's approach, produce by far the longest "
                  "tidal tails."),
            ("h", "Units"),
            ("p", "The code works in units where G = 1, which keeps the numbers near one "
                  "and avoids wasting floating-point precision. Converting back:"),
            ("mono",
             "  length    1 unit  =  1 kpc           =  3.26 thousand light years\n"
             "  mass      1 unit  =  10^10 M_sun     =  10 billion suns\n"
             f"  velocity  1 unit  =  {VEL_UNIT_KMS} km/s\n"
             f"  time      1 unit  =  {TIME_UNIT_MYR} Myr        =  4.7 million years"),
            ("p", "For scale, the Sun orbits the Milky Way once every ~230 million years, "
                  "so the whole simulation below covers only a handful of galactic years."),
        ])

        # ---- montage ----
        picks = np.linspace(0, len(files) - 1, 6).astype(int)
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.08, 0.955, "3.  The encounter, start to finish", fontsize=19,
                 weight="bold", va="top")
        for k, idx in enumerate(picks):
            ax = fig.add_axes([0.08 + 0.29 * (k % 3), 0.60 - 0.28 * (k // 3), 0.27, 0.26])
            rgb = frame_rgb(np.fromfile(files[idx], np.float32).reshape(-1, 3),
                            masks, 500, args.half, sc_tile)
            ax.imshow(rgb, origin="lower", extent=[-args.half, args.half] * 2)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"t = {times[idx]:.2f} Gyr", fontsize=9)
        cap = ("Orange marks stars that began in the first galaxy, blue the second; "
               "the faint violet haze is dark matter. Watch for the bridge of material "
               "that forms between the two disks at closest approach, and the long tails "
               "flung outward on the opposite sides. Both are made of stars that were "
               "merely orbiting quietly beforehand - the companion's gravity pulls harder "
               "on the near side of a disk than on the far side, and that difference "
               "across the galaxy, the tidal force, is what tears the material out.")
        fig.text(0.08, 0.30, textwrap.fill(cap, 95), fontsize=10, va="top",
                 linespacing=1.6)
        pdf.savefig(fig); plt.close(fig)

        # ---- diagnostics ----
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.08, 0.955, "4.  Reading the merger from one number", fontsize=19,
                 weight="bold", va="top")
        ax = fig.add_axes([0.13, 0.56, 0.78, 0.30])
        ax.plot(times, sep, color="#c2410c", lw=2)
        ax.set_xlabel("time  [Gyr]"); ax.set_ylabel("separation of the two disks  [kpc]")
        ax.grid(alpha=0.3)
        ax.axvline(times[peak], color="0.5", ls="--", lw=1)
        ax.annotate(f"first pericentre\n{sep[peak]:.1f} kpc", (times[peak], sep[peak]),
                    textcoords="offset points", xytext=(16, 26), fontsize=9, color="0.3")
        txt = (f"The distance between the centres of the two stellar disks tells most of "
               f"the story. The galaxies fall together, reach {sep[peak]:.0f} kpc at first "
               f"pericentre around t = {times[peak]:.2f} Gyr, and swing apart again. "
               f"Each close passage converts orbital energy into the random motions of "
               f"individual stars - energy drained out of the orbit and into the tidal "
               f"tails and the thickening of the disks. Because that drain is one-way, "
               f"the galaxies cannot climb back to their original separation, so each "
               f"apocentre is lower than the last and the pair spirals in. This is "
               f"dynamical friction, and it is why galaxy encounters almost always end "
               f"in a merger rather than a clean flyby. The end state, a single pressure-"
               f"supported blob with no ordered rotation, is how astronomers think many "
               f"elliptical galaxies were made.")
        fig.text(0.08, 0.47, textwrap.fill(txt, 95), fontsize=10, va="top",
                 linespacing=1.6)
        fig.text(0.08, 0.20, "A caution about what this model leaves out", fontsize=12.5,
                 weight="bold", va="top")
        cav = ("There is no gas here, and therefore no star formation, no shocks and no "
               "feedback. Real colliding galaxies light up with newborn stars exactly "
               "where this simulation shows the densest tidal features, which is why the "
               "Antennae galaxies glow blue along their bridge. Treat these images as a "
               "map of where the mass goes, not a picture of what a telescope would see.")
        fig.text(0.08, 0.165, textwrap.fill(cav, 95), fontsize=10, va="top",
                 linespacing=1.6)
        pdf.savefig(fig); plt.close(fig)

        # ---- methods ----
        text_page(pdf, "5.  How it was computed", [
            ("h", "The machine"),
            ("p", "Aurora is an exascale supercomputer at Argonne National Laboratory. "
                  "Each node carries six Intel Data Center GPU Max 1550 accelerators, and "
                  "each of those is built from two tiles. With the frameworks module loaded "
                  "the runtime presents every tile as its own device, so a node offers "
                  "twelve. This run launched six MPI ranks, one per tile, and so used six "
                  "of the twelve available tiles - twelve ranks would have finished in "
                  "roughly half the time. Each rank holds a full copy of all particle "
                  "positions and computes the forces on its own contiguous slice. After "
                  "every step the slices are stitched back together with an MPI_Allgatherv, "
                  "so the forces are always evaluated against the complete system."),
            ("h", "Why direct summation"),
            ("p", "Most production N-body codes use a tree or a particle-mesh method to "
                  "avoid the N² cost, approximating distant groups of particles by "
                  "their combined mass. This code does not: it computes every pair "
                  "exactly. That is affordable on GPUs at this particle count, and it means "
                  "the only sources of error are the softening length and the timestep, "
                  "both of which are easy to reason about. The inner loop stages particles "
                  "through GPU local memory in tiles of 256 so that each one is fetched "
                  "once and reused across the whole tile."),
            ("h", "Measured performance"),
            ("mono", perf if perf else "(see sim.log)"),
            ("h", "Reproducing this"),
            ("mono",
             "  cd /lus/flare/projects/DD_plasma_physics/claude-galaxies\n"
             "  make                                   # build with mpicxx -fsycl\n"
             "  ./venv/bin/python make_ic.py --n 600000 --out ic.bin\n"
             "  qsub run_all.sh            # simulate, render, encode, report"),
            ("h", "Files"),
            ("mono",
             "  nbody.cpp      SYCL + MPI direct N-body solver\n"
             "  make_ic.py     builds the two galaxies and their orbit\n"
             "  render.py      snapshots -> PNG frames\n"
             "  report.py      this document\n"
             "  run_all.sh     the PBS job that produced everything here\n"
             "  config.sh      run parameters shared by the stages\n"
             "  collision.mp4  the animation"),
        ])

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
