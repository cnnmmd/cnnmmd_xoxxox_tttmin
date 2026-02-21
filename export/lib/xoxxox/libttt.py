from pathlib import Path
import torch
from transformers import GPT2LMHeadModel

class OpeTtt:

  def __init__(self, dirprm=None, chaunk=None):
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.chaunk = chaunk
    pthprm = Path(dirprm)
    dvocab = torch.load(pthprm / "dvocab.pt", map_location="cpu") # 文字の語彙
    self.cvstoi = dvocab["cvstoi"]
    self.cvitos = dvocab["cvitos"]
    self.nblock = dvocab["nblock"]
    self.keyunk = self.cvstoi[self.chaunk]
    self.nmodel = GPT2LMHeadModel.from_pretrained(pthprm) # 復元（モデル）
    self.nmodel.to(self.device)
    self.nmodel.eval()

  @torch.no_grad()
  def gentxt(self, prompt, maxtkn=100, numtmp=None, numtop=None, chaend=""):
    self.nmodel.eval()
    tsridx = OpeTtt.encode(prompt, self.cvstoi, self.keyunk).unsqueeze(0).to(self.device) # 変換（プロンプト）
    for _ in range(maxtkn):
      tsrcnd = tsridx[:, -self.nblock:] # 最大コンテキスト：nblock
      logits = self.nmodel(input_ids=tsrcnd).logits  # (1, T, V) # forward（labelsは渡さない）
      logits = logits[:, -1, :] # 最後の文字位置のみ
      if numtmp is not None:
        logits = logits / numtmp
      if numtop is not None:
        v, _ = torch.topk(logits, numtop)
        logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
      tsrprb = torch.softmax(logits, dim=-1)
      keynxt = torch.multinomial(tsrprb, num_samples=1)
      tsridx = torch.cat([tsridx, keynxt], dim=1)
      values, indice = torch.topk(tsrprb, 5) # 表示：次候補の文字、確率（スコア）
      result = []
      for p, i in zip(values[0], indice[0]):
        c = self.cvitos[int(i)]
        result.append((c, float(p)))
      #print(result) # DBG
      #print("") # DBG
      if chaend != "":
        if self.cvitos[int(keynxt)] in chaend:
          break
    return OpeTtt.decode(tsridx[0].cpu(), self.cvitos, self.chaunk) # プロンプト＋推定文字群

  @staticmethod
  def encode(str: str, cvstoi, keyunk) -> torch.Tensor:
    return torch.tensor([cvstoi.get(c, keyunk) for c in str], dtype=torch.long)

  @staticmethod
  def decode(ids, cvitos, chaunk) -> str:
    if isinstance(ids, torch.Tensor):
      ids = ids.tolist()
    return "".join(cvitos.get(int(i), chaunk) for i in ids)
