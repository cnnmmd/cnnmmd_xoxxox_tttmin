import argparse
from xoxxox.shared import Custom
from xoxxox.libttt import OpeTtt

#---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("arg1")
parser.add_argument("--config")
parser.add_argument("--dirprm")
parser.add_argument("--chaunk")
parser.add_argument("--maxtkn", type=int)
parser.add_argument("--numtmp", type=float)
parser.add_argument("--numtop", type=float)
parser.add_argument("--chaend")
objarg = parser.parse_args()
dicprm = {k: v for k, v in vars(objarg).items() if v is not None}
diccnf = Custom.update(dicprm["config"], dicprm)

opettt= OpeTtt(dirprm=diccnf["dirprm"], chaunk=diccnf["chaunk"])
txtres = opettt.gentxt(objarg.arg1, maxtkn=diccnf["maxtkn"], numtmp=diccnf["numtmp"], numtop=diccnf["numtop"], chaend=diccnf["chaend"])
print(txtres)
