read "/home/duweiyue25/tgppo-transfer/research/tgppo/setup/scip-6.0.1"/check/"instances/MIP/stein27.fzn"
write problem temp/stein27.fzn.ppm
presolve
write transproblem temp/stein27.fzn_trans.ppm
read temp/stein27.fzn_trans.ppm
optimize
read temp/stein27.fzn.ppm
optimize
validatesolve "18" "18"
quit
