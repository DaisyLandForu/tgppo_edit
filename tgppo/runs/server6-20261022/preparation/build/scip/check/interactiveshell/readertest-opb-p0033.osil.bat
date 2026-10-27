read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/p0033.osil"
write problem temp/p0033.osil.opb
presolve
write transproblem temp/p0033.osil_trans.opb
read temp/p0033.osil_trans.opb
optimize
read temp/p0033.osil.opb
optimize
validatesolve "3089" "3089"
quit
