read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/lseu.mps"
write problem temp/lseu.mps.gms
presolve
write transproblem temp/lseu.mps_trans.gms
read temp/lseu.mps_trans.gms
optimize
read temp/lseu.mps.gms
optimize
validatesolve "1120" "1120"
quit
