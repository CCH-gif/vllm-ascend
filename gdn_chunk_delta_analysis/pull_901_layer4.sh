#!/bin/bash
# Resumable download of CANN 9.0.1 (arm64) layer4 (the CANN toolkit layer) from
# swr.cn-south-1.myhuaweicloud.com/ascendhub/cann  (anonymous bearer token)
# Usage: bash pull_901_layer4.sh > /tmp/cann901/pull_901.log 2>&1
set -u
DIR=/tmp/cann901
BLOB=$DIR/layer4.tar.gz
DIGEST=sha256:6318ce44394d68598b78d9f5a691ab8f9540794db0be513c319aaad36f3faaa7
WANT=4116426608
TOK_URL='https://swr.cn-south-1.myhuaweicloud.com/swr/auth/v2/registry/auth/?service=dockyard&scope=repository:ascendhub/cann:pull'
URL="https://swr.cn-south-1.myhuaweicloud.com/v2/ascendhub/cann/blobs/$DIGEST"

gettoken() {
  curl -s "$TOK_URL" | /usr/bin/python3 -c "import sys,json;print(json.load(sys.stdin).get('token',''))"
}

mkdir -p "$DIR"
n=0
while :; do
  n=$((n+1))
  got=$(stat -c%s "$BLOB" 2>/dev/null || echo 0)
  if [ "$got" -ge "$WANT" ]; then echo "DONE size=$got"; break; fi
  TOK=$(gettoken)
  [ -n "$TOK" ] || { echo "no token attempt=$n"; sleep 5; continue; }
  code=$(curl -s -C - -H "Authorization: Bearer $TOK" -o "$BLOB" -w "%{http_code}" -L "$URL")
  got=$(stat -c%s "$BLOB" 2>/dev/null || echo 0)
  echo "attempt=$n code=$code got=$got want=$WANT"
  if [ "$got" -ge "$WANT" ]; then echo "DONE size=$got"; break; fi
  if [ "$n" -ge 2000 ]; then echo "GIVE_UP attempt=$n"; break; fi
  # 416 = resume range not satisfiable -> restart from 0 is wrong; wait & recheck size
  if [ "$code" = "416" ]; then echo "416 seen (file may be complete but <want?)"; fi
  sleep 2
done
echo "FINISHED n=$n"
