# HF 常驻推理服务(新文件):模型加载一次,TCP 127.0.0.1:51765 接单。
# 协议: 一行 "csv路径|输出wav路径" -> 回 "OK"/"ERR"。logmel_v3_copy 经 runpy 原样调用(不改动)。
import os, sys, runpy, socketserver, traceback
import numpy as np, csv as csvmod
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import torch, soundfile as sf
from pathlib import Path
from nsf_hifigan import NsfHifiGAN
model = NsfHifiGAN(Path("model.ckpt")); model.to_device("cpu")
SR = model.h.sampling_rate
def make_f0(csvp):
    rows=[]
    with open(csvp,newline="") as f:
        r=csvmod.reader(f); next(r)
        for row in r:
            if len(row)>=2: rows.append(float(row[1]))
    return np.asarray(rows,dtype=np.float32)
def render(csvp,outp):
    bak=sys.argv
    sys.argv=["logmel_v3_copy.py","--ref_csv",csvp,"--out_npy","input.npy"]
    try:
        try: runpy.run_path("logmel_v3_copy.py",run_name="__main__")
        except SystemExit as e:
            if e.code not in (0,None): raise
    finally: sys.argv=bak
    mel=np.load("input.npy"); f0=make_f0(csvp)
    T=min(len(f0),mel.shape[0])
    with torch.no_grad():
        wav=model.model(torch.from_numpy(mel[:T].T).unsqueeze(0).float(),
                        torch.from_numpy(f0[:T]).unsqueeze(0).float()).view(-1)
    sf.write(outp,wav.cpu().numpy(),SR,subtype="PCM_16")
class H(socketserver.StreamRequestHandler):
    def handle(self):
        line=self.rfile.readline().decode("mbcs","replace").strip()
        try:
            csvp,outp=line.split("|",1)
            render(csvp,outp)
            self.wfile.write(b"OK\n")
        except Exception:
            traceback.print_exc()
            self.wfile.write(b"ERR\n")
socketserver.TCPServer.allow_reuse_address=True
srv=socketserver.TCPServer(("127.0.0.1",51765),H)
print("HF daemon ready",flush=True)
srv.serve_forever()
