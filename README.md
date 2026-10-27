# Two Galaxies Colliding on Aurora

A direct-summation N-body simulation of two disk galaxies merging, written in
SYCL + MPI and run on [Aurora](https://www.alcf.anl.gov/aurora), the exascale
system at Argonne National Laboratory.

No tree code, no particle mesh, no multipole approximation: every pair of
particles interacts with every other pair at every timestep. At 600,000
particles that is 3.6 × 10¹¹ interactions per step, which is affordable on
Aurora's Intel Data Center GPU Max 1550 accelerators and keeps the error
budget down to two things that are easy to reason about — the softening
length and the timestep.

## Results

| | |
|---|---|
| [`results/collision.mp4`](results/collision.mp4) | 445-frame animation, 1080×1080, ~15 s |
| [`results/galaxy_collision_report.pdf`](results/galaxy_collision_report.pdf) | 6-page explainer for an undergraduate STEM reader |
| [`results/sim.log`](results/sim.log) | full simulation output |

Orange marks stars that began in the first galaxy, blue the second, and the
faint violet haze is dark matter.

## What happens

The galaxies start 60 kpc apart on a parabolic orbit — the marginally-bound
case, two galaxies falling together from rest at infinity.

| Phase | Time | Separation |
|---|---|---|
| Start | 0.00 Gyr | 60.0 kpc |
| First pericentre | 0.14 Gyr | 16.4 kpc |
| Apocentre | ~0.44 Gyr | 48.8 kpc |
| Second pericentre | ~0.75 Gyr | 5.6 kpc |
| Merged | ~0.95 Gyr | < 1 kpc |

The apocentre comes back to 48.8 kpc rather than the original 60. That gap is
dynamical friction: each close passage drains orbital energy into tidal tails
and into the random motions of individual stars, and because the drain is
one-way the pair cannot climb back out. It is why galaxy encounters almost
always end in a merger rather than a clean flyby.

Tidal material reaches 240 kpc from the centre of mass — about 2.5 times the
width of the rendered frame.

## Method

- **Gravity** — direct summation with Plummer softening, ε = 0.08 kpc. The
  `j = i` term vanishes because its numerator is the zero vector, so no branch
  is needed to skip self-interaction.
- **Integration** — kick-drift-kick leapfrog. Symplectic, so energy wanders
  but does not drift over tens of thousands of steps.
- **Parallelism** — one MPI rank per GPU tile. Each rank keeps a full copy of
  all positions and computes forces for a contiguous slice, then the slices
  are stitched back with `MPI_Allgatherv`. The force kernel stages source
  particles through GPU local memory in tiles of 256.
- **Initial conditions** — each galaxy is a Plummer sphere (halo + bulge,
  sampled exactly from its isotropic distribution function) plus an
  exponential stellar disk on near-circular orbits.

### Units

The code works in units where G = 1:

| Quantity | 1 unit |
|---|---|
| length | 1 kpc |
| mass | 10¹⁰ M☉ |
| velocity | 207.4 km/s |
| time | 4.715 Myr |

## Performance

From PBS job 8971942, one Aurora node:

```
done: 20000 steps in 1977.4 s  (98.87 ms/step, 3.64e+12 interactions/s)
```

A caveat worth recording: that run launched 6 MPI ranks on a node that
exposes **12** GPU tiles, so it used half the available hardware. Each Max
1550 is built from two tiles, and with the `frameworks` module loaded the
runtime presents each tile as its own device. Launch with `-n 12 -ppn 12` to
use the whole node.

## Running it

On an Aurora login node:

```bash
make                                              # mpicxx -fsycl
python3 -m venv venv && ./venv/bin/pip install numpy matplotlib
./venv/bin/python make_ic.py --n 600000 --out ic.bin
qsub run_all.sh                                   # simulate, render, encode, report
```

`run_all.sh` is deliberately a single self-contained job rather than a
`qsub` dependency chain: Aurora's `debug` queue permits only one queued job
per user, and keeping everything in one job means nothing has to resubmit on
your behalf if you disconnect.

Note that `debug` caps at 2 nodes and 1 hour, while every production queue
(`prod`, `small`, `medium`, `large`) has a **256-node minimum**. There is no
queue for a long single-node job.

## Layout

```
nbody.cpp       SYCL + MPI direct N-body solver
make_ic.py      builds the two galaxies and their encounter orbit
render.py       snapshots -> PNG frames
report.py       generates the PDF
run_all.sh      the PBS job that produced everything in results/
config.sh       run parameters shared across stages
examples/       a minimal SYCL device-enumeration + SAXPY benchmark
```

## What this model leaves out

There is no gas, so no star formation, no shocks and no feedback. Real
colliding galaxies light up with newborn stars exactly where this simulation
shows the densest tidal features — which is why the Antennae galaxies glow
blue along their bridge. These images map where the mass goes, not what a
telescope would see.
