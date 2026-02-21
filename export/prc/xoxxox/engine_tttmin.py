from xoxxox.shared import Custom
from xoxxox.libttt import OpeTtt

#---------------------------------------------------------------------------

class TttPrc:

  def __init__(self, config="xoxxox/config_tttmin_000", **dicprm):
    diccnf = Custom.update(config, dicprm)
    
  def status(self, config="xoxxox/config_tttmin_000", **dicprm):
    diccnf = Custom.update(config, dicprm)
    self.opettt= OpeTtt(dirprm=diccnf["dirprm"], chaunk=diccnf["chaunk"])
    self.maxtkn = diccnf["maxtkn"]
    self.numtmp = diccnf["numtmp"]
    self.numtop = diccnf["numtop"]
    self.chaend = diccnf["chaend"]

  def infere(self, txtreq):
    txtres = self.opettt.gentxt(txtreq, maxtkn=self.maxtkn, numtmp=self.numtmp, numtop=self.numtop, chaend=self.chaend)
    return (txtres,"")
