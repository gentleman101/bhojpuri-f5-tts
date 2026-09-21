"""Dev check: does torch.compile of the transformer forward match eager? Same weights, same batches, same seeds.

Run:  python scripts/dev/check_compile_equivalence.py      (needs the GPU, ~6 min, mostly compile time)
Result on 2026-09-21: loss within ~0.5% on 6 batches; gradient cosine 0.75-0.98 (eager vs eager: 0.99999). Accepted after a matched
500-update training run gave the same validation loss (0.6744 vs 0.6749).
LESSON: F5 draws its condition dropout from Python's `random`, so seed random + numpy + torch, or eager-vs-eager will disagree too.
"""
import torch, yaml, math, random, numpy as np
from bhojpuri_tts import REPO_ROOT
from bhojpuri_tts.data import FrameBatchSampler, MelDataset, collate, read_manifest
from bhojpuri_tts.modeling import add_lora, build_cfm, load_base_weights
cfg=yaml.safe_load(open(REPO_ROOT/"configs/lora_slice.yaml"))
m=cfg["model"]; model=build_cfm(str(REPO_ROOT/m["vocab_file"]), m["arch"]); load_base_weights(model, str(REPO_ROOT/m["base_checkpoint"]))
add_lora(model, cfg["lora"])
torch.manual_seed(123)
with torch.no_grad():
    for n,p in model.named_parameters():
        if p.requires_grad and "lora_B" in n: p.normal_(0,0.02)   # make the LoRA path matter (B starts at 0)
model.cuda().train()
params=[p for p in model.parameters() if p.requires_grad]
print("trainable tensors:",len(params), "| with nonzero B:", sum(1 for n,p in model.named_parameters() if p.requires_grad and "lora_B" in n and p.abs().sum()>0))
rows=read_manifest(REPO_ROOT/cfg["data"]["train_manifest"]); ds=MelDataset(rows)
samp=FrameBatchSampler([ds.frame_len(i) for i in range(len(ds))], seed=cfg["seed"], max_frames=cfg["data"]["max_frames_per_batch"], max_samples=cfg["data"]["max_samples_per_batch"])
batches=[collate([ds[i] for i in idx]) for _,idx in zip(range(6), iter(samp))]
def step(b, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)  # F5 uses python random for cond dropout
    for p in params: p.grad=None
    with torch.autocast("cuda",dtype=torch.bfloat16):
        loss,_,_=model(b["mel"].cuda(), text=b["text"], lens=b["lens"].cuda())
    loss.backward()
    g=torch.cat([p.grad.flatten().float() for p in params if p.grad is not None])
    return loss.item(), g
eager=[step(b,i) for i,b in enumerate(batches)]
eager2=[step(b,i) for i,b in enumerate(batches)]
model.transformer.forward=torch.compile(model.transformer.forward, dynamic=True)
comp=[]
for i,b in enumerate(batches):
    comp.append(step(b,i)); print("compiled batch",i,"done",flush=True)
cos=lambda a,b: float(torch.nn.functional.cosine_similarity(a,b,dim=0))
print("\nbatch | loss eager  eager2  compiled | grad-cos(eager,eager2)  grad-cos(eager,compiled) | |g| eager  compiled")
for i in range(len(batches)):
    (l0,g0),(l1,g1),(l2,g2)=eager[i],eager2[i],comp[i]
    print(f"{i:5} | {l0:.4f}  {l1:.4f}  {l2:.4f}   |  {cos(g0,g1):.5f}   {cos(g0,g2):.5f}   | {g0.norm():.4f}  {g2.norm():.4f}")
print("EQUIV_DONE")
