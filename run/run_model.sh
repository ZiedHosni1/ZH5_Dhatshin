#!/bin/bash

for i in {1..5}; do
  echo "Submitting run $i"
  qsub run_model.sge $i
done
