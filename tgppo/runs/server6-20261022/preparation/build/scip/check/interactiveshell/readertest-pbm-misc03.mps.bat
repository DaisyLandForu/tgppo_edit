read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/misc03.mps"
write problem temp/misc03.mps.pbm
presolve
write transproblem temp/misc03.mps_trans.pbm
read temp/misc03.mps_trans.pbm
optimize
read temp/misc03.mps.pbm
optimize
validatesolve "3360" "3360"
quit
