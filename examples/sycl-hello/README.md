# claude-sycle-hello

Minimal SYCL example for Aurora: enumerates the GPU devices, then runs a
SAXPY (`y = 2.5*x + y`) over 16.7M floats on one device and reports achieved
bandwidth.

## Build

On a login node (UAN), the oneAPI compiler is already on `PATH`:

    make

## Run

    qsub submit.sh

Or interactively:

    qsub -I -A DD_plasma_physics -q debug -l select=1 \
         -l walltime=00:05:00 -l filesystems=home:flare
    cd ~/claude-sycle-hello && ./vecadd

## Measured

Job 8971856, 1 node, 1 second walltime:

    SAXPY  n=16777216  time=0.106681 ms  1887.18 GB/s
    PASS  (0 mismatches)

## Notes

- Each Max 1550 has 2 tiles. How they are exposed depends on
  `ZE_FLAT_DEVICE_HIERARCHY`:
  - `COMPOSITE` - 6 devices per node, each a whole 2-tile GPU
    (896 CUs, ~128 GB HBM). **This is what the run above saw.**
  - `FLAT` - 12 devices per node, one per tile (448 CUs, ~64 GB each).

  The device count and the reported CU / memory figures tell you which mode
  you are in.
- Under `COMPOSITE`, a single in-order queue drives only one tile unless you
  enable implicit scaling or split the device into sub-devices. The bandwidth
  above is therefore roughly half of what a full GPU can sustain.
- Login nodes have no GPUs, so `./vecadd` fails the `gpu_selector_v` lookup
  there. It must run on a compute node.
