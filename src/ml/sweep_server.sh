srun \
--export=ALL \
--container-image=./bdalt+future-statements+training+latest.sqsh \
--container-mounts=/mnt/ceph:/mnt/ceph \
bash -c " cd $WORKDIR && PYTHONPATH=. python3 $SWEEP_SERVER" \
| tee >&2 >(tail -1)