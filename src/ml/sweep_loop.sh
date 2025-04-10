#!/bin/bash
for i in $(seq $NWORKERS);
        do sbatch sweep_agent.sh $i;
done