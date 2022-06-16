#!/bin/bash
#$ -wd /well/nichols/users/pra123/CBMR
#$ -cwd
#$ -P nichols.prjc
#$ -N QPoi
#$ -q short.qg
#$ -o logs/test.out 
#$ -e logs/test.err 
#$ -t 2-21:1
#$ -tc 10
#$ -l gpu=1
#$ -l m_mem_free=10G

source ~/.bashrc 
cd CBMR
source /well/nichols/users/pra123/anaconda3/bin/activate torch


SECONDS=0
echo $(date +%d/%m/%Y\ %H:%M:%S)

cmdList="$1"
cmd=$(sed -n ${SGE_TASK_ID}p $cmdList)
echo "$cmd"
bash -c "$cmd"

duration=$SECONDS
echo "CPU time $pheno: $(($duration / 60)) min $((duration % 60)) sec"
echo $(date +%d/%m/%Y\ %H:%M:%S)

