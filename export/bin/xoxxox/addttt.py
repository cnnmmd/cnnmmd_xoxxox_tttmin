import argparse
from pathlib import Path
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from xoxxox.shared import Custom
from xoxxox.libttt import OpeTtt

#---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--config")
parser.add_argument("--pthsrc")
parser.add_argument("--numstp", type=int)
objarg = parser.parse_args()
dicprm = {k: v for k, v in vars(objarg).items() if v is not None}
diccnf = Custom.update(dicprm["config"], dicprm)

dirprm = diccnf["dirprm"]
chaunk = diccnf["chaunk"]
pthsrc = diccnf["pthsrc"]
numstp = diccnf["numstp"]

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

p = Path(pthsrc)
with p.open(mode="r", encoding="utf-8") as f:
  txtsrc = f.read()

# 変換（文字〜数値）
lstcha = sorted(set(txtsrc))
lstcha.append(chaunk)
nvocab = len(lstcha)
cvstoi = {c:i for i,c in enumerate(lstcha)}
cvitos = {i:c for c,i in cvstoi.items()}
keyunk = cvstoi[chaunk]

datsrc = OpeTtt.encode(txtsrc, cvstoi, keyunk)
n = int(0.9 * len(datsrc))
dattrn, datval = datsrc[:n], datsrc[n:]

nblock=64
nbatch=64

# バッチ
def getbch(csplit):
  src = dattrn if csplit=="trn" else datval
  idx = torch.randint(0, len(src) - nblock - 1, (nbatch,))
  x = torch.stack([src[i:i+nblock] for i in idx])
  y = torch.stack([src[i+1:i+nblock+1] for i in idx])
  return x.to(device), y.to(device)

# 定義（モデル）
config = GPT2Config(
  vocab_size=nvocab,
  n_positions=nblock,
  n_ctx=nblock,
  n_embd=128,
  n_layer=4,
  n_head=4,
)
config.loss_type="ForCausalLMLoss"
nmodel = GPT2LMHeadModel(config).to(device)
opt = torch.optim.AdamW(nmodel.parameters(), lr=3e-4)

# 表示（ライン／ＣＬＩ）
def drwlin(length, minval, maxval, nvalue=None):
  lin = ["-"] * int(length)
  if nvalue is not None and maxval > minval:
    pos = int((nvalue - minval) / (maxval - minval) * (length - 1))
    pos = max(0, min(length - 1, pos))
    lin[pos] = "+"
  return "".join(lin)

# 訓練
sml = 0
for stp in range(1, numstp):
  x, y = getbch("trn")
  out = nmodel(input_ids=x, labels=x) # out = nmodel(input_ids=x, labels=y)
  los = out.loss
  opt.zero_grad(set_to_none=True)
  los.backward()
  opt.step()
  # 表示（ロス）
  if stp % 1 == 0:
    #print(stp, float(los))
    if los <= 0.1:
      sml = 1
    print(f"{stp:4} {los:2.8f} ", end='')
    if sml == 0:
      print(drwlin(60, 0, 5, nvalue=los))
    else:
      print(drwlin(60, 0, 0.1, nvalue=los))

# 保存
pthprm = Path(dirprm)
pthprm.mkdir(parents=True, exist_ok=True)
nmodel.save_pretrained(pthprm) # 基本
torch.save({"cvstoi": cvstoi, "cvitos": cvitos, "nblock": nblock}, pthprm / "dvocab.pt") # 文字の語彙
torch.save( # 訓練（再開向け）
  {
    "step": stp,
    "optimizer": opt.state_dict(),
    "model_state_dict": nmodel.state_dict(),
  },
  pthprm / "status.pt",
)
