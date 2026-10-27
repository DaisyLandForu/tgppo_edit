read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/vpm2.fzn"
write problem temp/vpm2.fzn.pip
presolve
write transproblem temp/vpm2.fzn_trans.pip
read temp/vpm2.fzn_trans.pip
optimize
read temp/vpm2.fzn.pip
optimize
validatesolve "13.75" "13.75"
quit
