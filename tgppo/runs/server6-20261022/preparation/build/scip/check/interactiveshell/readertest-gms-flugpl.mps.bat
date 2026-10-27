read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/flugpl.mps"
write problem temp/flugpl.mps.gms
presolve
write transproblem temp/flugpl.mps_trans.gms
read temp/flugpl.mps_trans.gms
optimize
read temp/flugpl.mps.gms
optimize
validatesolve "1201500" "1201500"
quit
