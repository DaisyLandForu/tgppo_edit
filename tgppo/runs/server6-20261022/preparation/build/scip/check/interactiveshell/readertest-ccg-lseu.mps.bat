read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/lseu.mps"
write problem temp/lseu.mps.ccg
presolve
write transproblem temp/lseu.mps_trans.ccg
read temp/lseu.mps_trans.ccg
optimize
read temp/lseu.mps.ccg
optimize
validatesolve "1120" "1120"
quit
