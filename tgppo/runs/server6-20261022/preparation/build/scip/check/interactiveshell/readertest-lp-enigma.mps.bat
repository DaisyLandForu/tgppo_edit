read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/enigma.mps"
write problem temp/enigma.mps.lp
presolve
write transproblem temp/enigma.mps_trans.lp
read temp/enigma.mps_trans.lp
optimize
read temp/enigma.mps.lp
optimize
validatesolve "0" "0"
quit
