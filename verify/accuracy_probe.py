"""精度 A/B：固定 token-id prompt，贪心解码，比对 logprob。

对同一批请求分别打到 base(AscendC) 和 triton 两个 server，比较
每个位置选中 token 的 logprob。logprob 是连续量，比 token 是否相同敏感得多——
LoRA 数值上任何真实差异都会在这里显形，而 token 相等可能只是没翻转。

用法: accuracy_probe.py <label> <out.json> [--lora NAME] [--n 64] [--len 4096]
"""
import argparse, json, os, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("label"); p.add_argument("out")
p.add_argument("--url", default="http://localhost:8001")
# 基座模型名（不带 LoRA 时打的那个）。与 /tmp/e2e/model.sh 的 SERVED_NAME 一致。
p.add_argument("--model", default=os.environ.get("SERVED_NAME", "Qwen36"))
p.add_argument("--lora", default=None, help="走 LoRA 的模型名；不给则全走 base")
p.add_argument("--n", type=int, default=64)
p.add_argument("--len", type=int, default=4096)
p.add_argument("--out-tokens", type=int, default=8)
p.add_argument("--concurrency", type=int, default=32)
p.add_argument("--seed", type=int, default=1234)
p.add_argument("--vocab", type=int, default=None,
               help="prompt token id 上界；默认从 $MODEL_PATH/config.json 读 vocab_size")
a = p.parse_args()


def _model_vocab():
    """从 $MODEL_PATH 的 config.json 读词表大小。

    以前硬编码 248320（Qwen3.5-9B 的词表）。换到 Qwen3-30B-A3B（151936）后
    生成的 token id 会越界，embedding 查表直接出错。读配置文件即可两个模型都对。
    """
    mp = os.environ.get("MODEL_PATH")
    if not mp:
        return None
    try:
        with open(os.path.join(mp, "config.json")) as fh:
            c = json.load(fh)
        v = c.get("vocab_size")
        if v is None:
            v = (c.get("text_config") or {}).get("vocab_size")
        return v
    except Exception:
        return None


rng = random.Random(a.seed)
VOCAB = a.vocab or _model_vocab() or 248320
print(f"[probe] vocab={VOCAB} (MODEL_PATH={os.environ.get('MODEL_PATH')})")
# 固定 prompt（token id），两个 server 用完全相同的输入
prompts = [[rng.randrange(1000, VOCAB) for _ in range(a.len)] for _ in range(a.n)]

def one(i):
    body = {
        "model": a.lora or a.model,
        "prompt": prompts[i],
        "max_tokens": a.out_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "logprobs": 1,
        # without this the completions API returns no token ids at all and the
        # comparison silently runs over zero tokens (choices[0] has no
        # "token_ids" key; the ids only appear when asked for).
        "return_token_ids": True,
    }
    data = json.dumps(body).encode()
    # 偶发 400 实测存在（约 64 个请求里 2~3 个），服务端只记 "400 Bad Request"
    # 不记原因。重试并把 body 打出来，否则整点会因为 ex.map 立刻抛出而全丢。
    last = None
    for attempt in range(6):
        req = urllib.request.Request(
            a.url + "/v1/completions", data=data,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return i, json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:400]
            last = f"HTTP {e.code} body={detail}"
            if attempt == 0:
                print(f"  [req {i}] {last}", flush=True)
        except Exception as e:  # noqa: BLE001 - 网络层抖动同样重试
            last = f"{type(e).__name__}: {e}"
            if attempt == 0:
                print(f"  [req {i}] {last}", flush=True)
        time.sleep(0.5 * (attempt + 1))
    return i, {"__error__": last}


out, errors = {}, {}
with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
    for i, resp in ex.map(one, range(a.n)):
        if "__error__" in resp:
            errors[i] = resp["__error__"]
            continue
        ch = resp["choices"][0]
        lp = ch.get("logprobs") or {}
        out[str(i)] = {
            # token_ids when the server honours return_token_ids, else the
            # decoded strings -- either is a fine identity for a token.
            "tokens": ch.get("token_ids") or lp.get("tokens") or [],
            "logprobs": [v for v in (lp.get("token_logprobs") or [])],
        }
n_tok = sum(len(v["tokens"]) for v in out.values())
n_lp = sum(len(v["logprobs"]) for v in out.values())
if not n_tok or not n_lp:
    sys.exit(f"[{a.label}] FATAL: captured {n_tok} tokens / {n_lp} logprobs -- "
             "the comparison would be vacuous")
json.dump({"label": a.label, "lora": a.lora, "len": a.len, "n": a.n,
           "results": out}, open(a.out, "w"), indent=1)
print(f"[{a.label}] wrote {a.out}: {a.n} reqs, {n_tok} tokens, {n_lp} logprobs, "
      f"lora={a.lora}, prompt_len={a.len}")
