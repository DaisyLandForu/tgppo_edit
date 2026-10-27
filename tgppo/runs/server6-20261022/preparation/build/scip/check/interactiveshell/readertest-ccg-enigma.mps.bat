read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/enigma.mps"
write problem temp/enigma.mps.ccg
presolve
write transproblem temp/enigma.mps_trans.ccg
read temp/enigma.mps_trans.ccg
optimize
read temp/enigma.mps.ccg
optimize
validatesolve "0" "0"
quit
