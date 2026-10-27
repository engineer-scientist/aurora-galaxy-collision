#!/bin/bash
#PBS -N gal-all
#PBS -A DD_plasma_physics
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o out/run_all.log

# Simulation and post-processing in one job. The debug queue allows only one
# job per user in Q state, so a qsub dependency chain is not an option; and
# keeping it in one job means the whole pipeline survives a disconnect without
# anything needing to resubmit on our behalf.
#
# Budget at 97.7 ms/step (measured, job 8971931):
#   sim    20000 steps  ~33 min
#   post   render+encode+report ~6 min
#   total  ~39 min against a 60 min wall

cd "$PBS_O_WORKDIR" || exit 1
source ./config.sh
mkdir -p snaps frames out artifacts

rm -f artifacts/PIPELINE_DONE artifacts/PIPELINE_FAILED

echo "=== gal-all start $(date) ==="
echo "N=$NPART steps=$STEPS dt=$DT snap_every=$SNAP_EVERY eps=$EPS"

module load frameworks >/dev/null 2>&1
mpiexec -n 6 -ppn 6 ./nbody \
    --ic ic.bin --out snaps \
    --dt "$DT" --steps "$STEPS" --snap-every "$SNAP_EVERY" --eps "$EPS" \
    2>&1 | tee out/sim.log
SIM_RC=${PIPESTATUS[0]}
echo "=== sim rc=$SIM_RC  $(date) ==="

if [ "$SIM_RC" -ne 0 ]; then
    echo "simulation failed, stopping" | tee artifacts/PIPELINE_FAILED
    exit 1
fi

# Each stage is attempted even if an earlier one failed, so a broken encoder
# does not cost us the report.
echo "=== render $(date) ==="
./venv/bin/python render.py --snaps snaps --ic ic.bin --out frames \
    --px 1080 --half "$HALF" --dt "$DT" --snap-every "$SNAP_EVERY" \
    || echo "RENDER FAILED"

echo "=== encode $(date) ==="
module load ffmpeg >/dev/null 2>&1
ffmpeg -y -loglevel warning -framerate 30 -i frames/frame_%05d.png \
    -c:v libx264 -pix_fmt yuv420p -crf 18 -movflags +faststart \
    artifacts/collision.mp4 || echo "ENCODE FAILED"

echo "=== report $(date) ==="
./venv/bin/python report.py --snaps snaps --ic ic.bin --simlog out/sim.log \
    --out artifacts/galaxy_collision_report.pdf \
    --dt "$DT" --snap-every "$SNAP_EVERY" --half "$HALF" \
    || echo "REPORT FAILED"

cp -f nbody.cpp make_ic.py render.py report.py config.sh Makefile run_all.sh \
      artifacts/ 2>/dev/null
cp -f out/sim.log artifacts/ 2>/dev/null

echo "=== artifacts ==="
ls -la artifacts/
date > artifacts/PIPELINE_DONE
echo "=== gal-all done $(date) ==="
