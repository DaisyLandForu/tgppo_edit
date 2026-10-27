read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/misc03.mps"
write problem temp/misc03.mps.pip
presolve
write transproblem temp/misc03.mps_trans.pip
read temp/misc03.mps_trans.pip
optimize
read temp/misc03.mps.pip
optimize
validatesolve "3360" "3360"
quit
