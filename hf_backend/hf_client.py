# HF 轻客户端(不 import torch,启动快):提交任务;daemon 不在则拉起并等待就绪。
import socket, sys, time, subprocess, os
HERE=os.path.dirname(os.path.abspath(__file__))
def try_send(csvp,outp,timeout):
    s=socket.create_connection(("127.0.0.1",51765),timeout=3)
    s.settimeout(timeout)
    s.sendall((csvp+"|"+outp+"\n").encode("mbcs"))
    r=s.makefile().readline().strip()
    s.close(); return r=="OK"
csvp,outp=sys.argv[1],sys.argv[2]
try:
    ok=try_send(csvp,outp,300); sys.exit(0 if ok else 1)
except (ConnectionRefusedError,OSError):
    pass
subprocess.Popen([sys.executable,os.path.join(HERE,"hf_daemon.py")],cwd=HERE,
    creationflags=0x08000208 if False else 0x00000008)  # DETACHED_PROCESS
deadline=time.time()+180
while time.time()<deadline:
    try:
        ok=try_send(csvp,outp,300); sys.exit(0 if ok else 1)
    except (ConnectionRefusedError,OSError):
        time.sleep(1.0)
print("daemon start timeout",file=sys.stderr); sys.exit(1)
