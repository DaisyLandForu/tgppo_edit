read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/enigma.mps"
write problem temp/enigma.mps.pbm
presolve
write transproblem temp/enigma.mps_trans.pbm
read temp/enigma.mps_trans.pbm
optimize
read temp/enigma.mps.pbm
optimize
validatesolve "0" "0"
quit
