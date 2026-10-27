read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/stein27.fzn"
write problem temp/stein27.fzn.mps
presolve
write transproblem temp/stein27.fzn_trans.mps
read temp/stein27.fzn_trans.mps
optimize
read temp/stein27.fzn.mps
optimize
validatesolve "18" "18"
quit
