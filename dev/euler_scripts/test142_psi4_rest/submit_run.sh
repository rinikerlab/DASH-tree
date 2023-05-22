bsub -n 4 -W 24:00 -R "rusage[mem=4096,scratch=2000]" -J "q500[1-2000]" -e "./outerr/sub_init.err" -o "./outerr/sub_init.out" "./run_worker.sh \${LSB_JOBINDEX}"
