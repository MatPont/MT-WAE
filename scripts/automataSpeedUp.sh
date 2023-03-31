#!/bin/bash

lrMT=(0.05 0.0025 0.0025 0.1 0.025 0.1 0.001 0.025 0.005 0.01 0.01 0.025)
lrPD=(0.5 0.1 0.001 0.05 10 100 0.005 0.00025 0.0025 0.005 0.01 0.5)
names=(starting isabel sea vortex particular cloud astroT impact volcanic astro3D earthquake darkSky100)
lbsl=(10 10 5 10 10 10 10 10 10 10 10 10)
miMT=(1000 500 100 500 1500 500 500 500 500 100 500 500)
miPD=(1000 200 100 500 1000 500 500 500 500 100 500 500)
mai=(0 0 500 0 0 0 0 0 0 500 0 0)
ni=(4 4 1 4 4 4 4 4 4 4 4 4)
eiMT=(0 0 1 0 0 1 0 0 0 1 0 0)
eiPD=(0 0 0 0 0 1 0 0 0 0 0 0)
opc=(0 0 1 0 0 0 0 0 0 0 0 0)

bsl=10

if [ $# -lt 1 -o $# -gt 3 ]; then
  echo "Usage "$0 "numberOfThreads [persistenceThresholdMultiplier] [saveOutput]"
  exit
fi

noThreads=$1
ptMult=$2
if [[ $ptMult == "" ]]; then
  ptMult=1
fi
doSave=$3
if [[ $doSave == "" ]]; then
  doSave=0
fi

len=`expr ${#names[@]} - 1`
for nt in $noThreads 1; do
    for pd in 0 1; do
        for i in `seq 0 $len`; do
            lr=${lrMT[$i]}
            if [ $pd -eq 1 ]; then
                lr=${lrPD[$i]}
            fi
            mi=${miMT[$i]}
            if [ $pd -eq 1 ]; then
                mi=${miPD[$i]}
            fi
            ei=${eiMT[$i]}
            if [ $pd -eq 1 ]; then
                ei=${eiPD[$i]}
            fi
            dl=4
            : ' if [ $nt -eq 1 ]; then
                dl=5
            fi'
            command="python3 run.py -td ${names[$i]} -dl $dl -ds $doSave -pp dataRed -c 0.0 -lr $lr -nl 0 -iNG 3 -lNG 3 -af 1 -bsl $bsl -lBsl ${lbsl[$i]} -nt $nt -pd $pd -opc ${opc[$i]} -mi $mi -mai ${mai[$i]} -ei $ei -ni ${ni[$i]} -ptMult $ptMult"
            
            outFileName=${names[$i]}_PD_${pd}
            if [ $nt -eq 1 ]; then
                outFileName=${outFileName}_NT_$nt
            fi
            $command | tee $outFileName
            if [ -f $outFileName ]; then
                rm $outFileName
            fi
            
            : 'if [ $nt -eq 1 ]; then
                $command &
            else
                $command
            fi'
        done
    done
done
