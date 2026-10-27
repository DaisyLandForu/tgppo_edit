read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/vpm2.fzn"
write problem temp/vpm2.fzn.rlp
presolve
write transproblem temp/vpm2.fzn_trans.rlp
read temp/vpm2.fzn_trans.rlp
optimize
read temp/vpm2.fzn.rlp
optimize
validatesolve "13.75" "13.75"
quit
