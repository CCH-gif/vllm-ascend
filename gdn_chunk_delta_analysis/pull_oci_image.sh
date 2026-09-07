#!/bin/bash
# Generic anonymous OCI pull -> rootfs materialize (for CANN toolkit containers).
#
# Usage: pull_oci_image.sh <registry>/<repo>:<tag> [dest_dir]
#   examples:
#     quay.io/ascend/cann:9.1.0-910b-ubuntu22.04-py3.11                                     # CANN 9.1.0 (910b)
#     swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.1-910b-openeuler24.03-py3.12-devel # CANN 9.0.1 (910b)
#
# Env:
#   OCI_DRY=1      stop after token+manifest resolution (prints layer table, no downloads)
#   OCI_ARCH=arm64 platform to pick from a multi-arch index (default arm64)
#
# Steps: probe registry auth (anonymous; public repos need none) -> pull manifest/index
#   -> pick platform -> resumable per-blob download (sha256 verified) -> decompress+stack
#   extract into <dest>/rootfs -> print the CANN toolkit path under /usr/local/Ascend.
# Stdlib only (curl, tar, gzip, python3). Works on SWR / quay / any registry v2.
set -u

[ $# -lt 1 ] && { echo "usage: $0 <registry>/<repo>:<tag> [dest_dir]"; exit 2; }
IM="$1"
DEST="${2:-$(pwd)}"
REG="${IM%%/*}"
REST="${IM#*/}"                                  # repo[:tag]
if [[ "$REST" == *":"* ]]; then REPO="${REST%:*}"; TAG="${REST##*:}"; else REPO="$REST"; TAG="latest"; fi
ARCH="${OCI_ARCH:-arm64}"
BASE="https://$REG/v2/$REPO"
ROOTFS="$DEST/rootfs"
BLOBDIR="$DEST/blobs"
mkdir -p "$ROOTFS" "$BLOBDIR"
echo "== registry=$REG repo=$REPO tag=$TAG arch=$ARCH"

MTYPES="application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.docker.distribution.manifest.v2+json"
TOKEN=""                                         # empty = no auth needed

# ---- resolve auth: none (public 200) or anonymous bearer via 401 challenge ----
get_token() {
  local code hdr realm svc scope n v
  code=$(curl -s -o /dev/null -w "%{http_code}" -H "Accept: $MTYPES" "$BASE/manifests/$TAG")
  if [ "$code" = "200" ]; then
    TOKEN=""; echo "registry: no auth needed (public)"; return 0
  fi
  for n in 1 2 3; do
    hdr=$(curl -s -D - -o /dev/null -H "Accept: $MTYPES" "$BASE/manifests/$TAG" | tr -d '\r' | grep -i '^www-authenticate:' | head -1)
    realm=$(echo "$hdr" | sed -nE 's/.*realm="([^"]*)".*/\1/p')
    svc=$(  echo "$hdr" | sed -nE 's/.*service="([^"]*)".*/\1/p')
    scope=$(echo "$hdr" | sed -nE 's/.*scope="([^"]*)".*/\1/p')
    [ -n "$realm" ] || { sleep 2; continue; }
    TOKEN=$(curl -s --get "$realm" --data-urlencode "service=$svc" --data-urlencode "scope=$scope" \
      | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('token') or d.get('access_token') or '')")
    [ -n "$TOKEN" ] || { sleep 2; continue; }
    v=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $TOKEN" \
        -H "Accept: $MTYPES" "$BASE/manifests/$TAG")
    [ "$v" = "200" ] && { echo "token ok (anonymous bearer)"; return 0; }
    sleep 2
  done
  echo "FAILED to get anonymous token for $REG (last code=$v)"; exit 1
}
get_token
AUTH=(); [ -n "$TOKEN" ] && AUTH=(-H "Authorization: Bearer $TOKEN")

# ---- manifest / index resolution ----
IDX="$DEST/manifest.index.json"
curl -s "${AUTH[@]}" -H "Accept: $MTYPES" -o "$IDX" "$BASE/manifests/$TAG" || { echo "manifest fetch failed"; exit 1; }
python3 - "$IDX" "$DEST" "$BASE" "$TOKEN" "$ARCH" "$MTYPES" <<'PY'
import json,sys,urllib.request
idx,dest,base,token,arch,mtypes=sys.argv[1:]
d=json.load(open(idx))
headers={} if not token else {"Authorization":"Bearer "+token}
headers["Accept"]=mtypes
m=None
if "manifests" in d and d["manifests"]:                 # index / list -> pick arch
    pick=next((e for e in d["manifests"] if e.get("platform",{}).get("os")=="linux"
               and e.get("platform",{}).get("architecture") in (arch,"aarch64")),None)
    if not pick:
        pick=next((e for e in d["manifests"] if e.get("platform",{}).get("os")=="linux"),None) \
             or d["manifests"][0]
        print("!! no %s entry; using %s"%(arch,pick["digest"]),file=sys.stderr)
    mfn="%s/manifest.%s.json"%(dest,pick["digest"].split(":")[1][:12])
    req=urllib.request.Request(base+"/manifests/"+pick["digest"],headers=headers)
    m=json.load(urllib.request.urlopen(req))
    json.dump(m,open(mfn,"w"))
else:
    mfn="%s/manifest.json"%(dest,)
    json.dump(d,open(mfn,"w"))
    m=d
json.dump({"manifest_file":mfn,
           "layers":[{"digest":l["digest"],"size":l["size"],
                      "gzip":"+gzip" in l.get("mediaType","")} for l in m.get("layers",[])],
           "config":m.get("config",{}).get("digest")},
          open("%s/layers.json"%dest,"w"),indent=1)
print("manifest: %s  layers=%d"%(mfn,len(m.get("layers",[]))))
PY
layers="$DEST/layers.json"
if [ "${OCI_DRY:-0}" = "1" ]; then
  echo "DRY mode (would download+extract $(python3 -c "import json;print(len(json.load(open('$layers'))['layers']))") layers):"
  python3 -c "import json;L=json.load(open('$layers'))['layers'];[print('  %s  %10d  %s'%(l['digest'].split(':')[1][:16],l['size'],'gzip' if l['gzip'] else 'tar')) for l in L]"
  exit 0
fi

# ---- resumable blob download with sha256 verify ----
dl() {  # dl <sha256> -> returns after file complete+verified
  local d="$1" f="$BLOBDIR/$1" want n code got sum
  n=0
  while :; do
    n=$((n+1))
    auth=(); [ -n "$TOKEN" ] && auth=(-H "Authorization: Bearer $TOKEN")
    want=$(curl -s "${auth[@]}" -I "$BASE/blobs/sha256:$d" | tr -d '\r' | awk -F': ' 'tolower($1)=="content-length"{print $2}' | tail -1)
    code=$(curl -s -C - "${auth[@]}" -o "$f" -w "%{http_code}" -L "$BASE/blobs/sha256:$d")
    got=$(stat -c%s "$f" 2>/dev/null || echo 0)
    echo "  blob $d attempt $n code=$code got=$got want=$want"
    if [ -n "$want" ] && [ "$got" = "$want" ]; then
      sum=$(sha256sum "$f" | awk '{print $1}')
      if [ "$sum" = "$d" ]; then echo "  sha OK"; return 0; else echo "  sha MISMATCH, restart"; rm -f "$f"; fi
    fi
    sleep 2
    [ $n -ge 600 ] && { echo "GAVE UP $d"; return 1; }
  done
}
mapfile -t SHAS < <(python3 -c "import json;print('\n'.join(l['digest'].split(':')[1] for l in json.load(open('$layers'))['layers']))")
for s in "${SHAS[@]}"; do
  dl "$s" || { echo "abort on blob $s"; exit 1; }
done

# ---- stack-extract layers in order into rootfs ----
python3 - "$layers" "$BLOBDIR" "$ROOTFS" <<'PY'
import json,sys,os,glob,tarfile,gzip,shutil
layers=json.load(open(sys.argv[1]))["layers"]; blobs,root=sys.argv[2:]
def open_layer(f):
    """sniff gzip magic (1f 8b) rather than trust mediaType label."""
    with open(f,"rb") as fh: magic=fh.read(2)
    if magic==b"\x1f\x8b": return gzip.open(f,"rb"),"gzip"
    try:
        return open(f,"rb"),"tar"
    except OSError: pass
for i,l in enumerate(layers,1):
    sha=l["digest"].split(":")[1]; f=os.path.join(blobs,sha)
    if not (os.path.exists(f) and os.path.getsize(f)==l["size"]):
        print("missing/incomplete blob",sha); sys.exit(1)
    src,kind=open_layer(f)
    print("extract %d/%d %s (%s)"%(i,len(layers),l["digest"],kind))
    try:
        tf=tarfile.open(fileobj=src)
        members=[m for m in tf.getmembers() if not (m.name=="dev" or m.name.startswith("dev/"))]
        # honor OCI whiteouts before extraction
        for m in list(members):
            b=os.path.basename(m.name)
            if not b.startswith(".wh."): continue
            full=os.path.join(root,m.name)
            if b==".wh..wh..opq":
                for ex in glob.glob(os.path.join(os.path.dirname(full),"*")):
                    if os.path.isdir(ex) and not os.path.islink(ex): shutil.rmtree(ex,ignore_errors=True)
                    else:
                        try: os.unlink(ex)
                        except OSError: pass
            else:
                tgt=os.path.join(root,os.path.dirname(m.name),b[4:])
                if os.path.isdir(tgt) and not os.path.islink(tgt): shutil.rmtree(tgt,ignore_errors=True)
                else:
                    try: os.unlink(tgt)
                    except OSError: pass
            members.remove(m)
        tf.extractall(root,members=members,filter="data")
        tf.close()
    finally:
        src.close()
print("rootfs at",root)
PY
[ $? -ne 0 ] && { echo "extraction failed"; exit 1; }

# ---- report CANN toolkit location ----
echo "== toolkit dirs found =="
for t in "$ROOTFS"/usr/local/Ascend/cann-*; do
  [ -d "$t" ] || continue
  v=$(grep -m1 '^Version=' "$t/opp/version.info" 2>/dev/null || echo "?")
  echo "  $t   ($v)"
done
echo "example fingerprint command:"
echo "  python3 collect_op_fingerprint.py $ROOTFS/usr/local/Ascend/cann-9.1.0 ChunkGatedDeltaRule out.json"
