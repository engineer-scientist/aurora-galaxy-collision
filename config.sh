# Shared run parameters. Simulation and post-processing both read these so
# frame timestamps always match the integration.
#
# Timing is measured, not guessed: 97.7 ms/step for N=600k on one Aurora node
# (6 GPUs), from validation job 8971931. 24000 steps ~= 39 min.
DT=0.02             # timestep, in units of 4.715 Myr
STEPS=20000         # -> 20000*0.02*4.715 = 1886 Myr = 1.89 Gyr of evolution
SNAP_EVERY=45       # -> 445 snapshots -> ~15 s of video at 30 fps
EPS=0.08            # Plummer softening, kpc
NPART=600000
HALF=90             # image half-width, kpc
