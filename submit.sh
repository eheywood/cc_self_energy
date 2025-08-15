#!/bin/bash
#$ -cwd
#$ -pe smp 8
#$ -l s_rt=336:00:00

source ~/.bashrc

# Insert commands here
python cc_controller.py > cc_output.out


