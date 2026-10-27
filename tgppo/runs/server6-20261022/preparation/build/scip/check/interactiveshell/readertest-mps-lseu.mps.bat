read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/lseu.mps"
write problem temp/lseu.mps.mps
presolve
write transproblem temp/lseu.mps_trans.mps
read temp/lseu.mps_trans.mps
optimize
read temp/lseu.mps.mps
optimize
validatesolve "1120" "1120"
quit
