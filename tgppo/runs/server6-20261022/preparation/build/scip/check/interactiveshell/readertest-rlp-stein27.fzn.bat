read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/stein27.fzn"
write problem temp/stein27.fzn.rlp
presolve
write transproblem temp/stein27.fzn_trans.rlp
read temp/stein27.fzn_trans.rlp
optimize
read temp/stein27.fzn.rlp
optimize
validatesolve "18" "18"
quit
