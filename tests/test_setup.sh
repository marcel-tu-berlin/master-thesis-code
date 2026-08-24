#!/usr/bin/env bash
# Harness for setup.sh. Part 1 drives the whole script with stubbed uv, git and
# playwright, so nothing is installed and no clone is touched; part 2 exercises
# pin_clone against real throwaway git repos. Both run anywhere: `bash tests/test_setup.sh`.
REPO="$(cd "$(dirname "$0")/.." && pwd)"
W="$(mktemp -d)"
pass=0; fail=0
ok()   { pass=$((pass+1)); echo "PASS $1"; }
bad()  { fail=$((fail+1)); echo "FAIL $1"; }
chk()  { if eval "$2"; then ok "$1"; else bad "$1"; fi; }

########## part 1: whole script, stubbed tools ##########
mkdir -p "$W/repo" "$W/bin" "$W/elsewhere/.venv"
cp "$REPO/setup.sh" "$W/repo/"
cp "$REPO/requirements.lock.txt" "$W/repo/"
mkdir -p "$W/repo/pipeline"; cp "$REPO/pipeline/OPENENV_COMMIT" "$W/repo/pipeline/"
PIN="$(cat "$W/repo/pipeline/OPENENV_COMMIT")"

cat > "$W/bin/uv" <<EOF
#!/usr/bin/env bash
echo "\$@" >> "$W/uv.log"
if [ "\$1" = "venv" ]; then mkdir -p "\$2/bin"; echo ':' > "\$2/bin/activate"; fi
exit 0
EOF
cat > "$W/bin/playwright" <<EOF
#!/usr/bin/env bash
echo "\$@" >> "$W/playwright.log"
EOF
cat > "$W/bin/git" <<EOF
#!/usr/bin/env bash
echo "\$@" >> "$W/git.log"
key="\$(echo "\$2" | tr / _)"
case "\$3" in
  cat-file)  exit 0 ;;                              # pin already present
  status)    exit 0 ;;                              # clean
  rev-parse) cat "$W/head_\$key" 2>/dev/null || echo before ;;
  checkout)  echo "\$5" > "$W/head_\$key" ;;        # checkout --quiet <sha>
esac
exit 0
EOF
chmod +x "$W/bin"/*
mkdir -p "$W/openenv_stub" "$W/miniwob_stub"

( cd "$W/elsewhere" && PATH="$W/bin:$PATH" VIRTUAL_ENV="$W/elsewhere/.venv" \
    OPENENV_DIR="$W/openenv_stub" MINIWOB_DIR="$W/miniwob_stub" \
    bash "$W/repo/setup.sh" > "$W/out.log" 2>&1 )
rc=$?
chk "runs to completion from a foreign cwd with VIRTUAL_ENV set (rc=$rc)" "[ $rc -eq 0 ]"
chk "caller's cwd venv untouched"        "[ -d '$W/elsewhere/.venv' ]"
chk "venv created in the repo, not cwd"  "[ -d '$W/repo/.venv' ]"
chk "interpreter pinned to 3.12.3"       "grep -q -- 'venv .venv --python 3.12.3' '$W/uv.log'"
chk "installs from the lock with cu130"  "grep -q -- 'pip install -r requirements.lock.txt --torch-backend=cu130' '$W/uv.log'"
chk "editable openenv install is --no-deps" "grep -q -- 'pip install --no-deps -e $W/openenv_stub' '$W/uv.log'"
chk "openenv-core uninstalled first"     "grep -q 'pip uninstall openenv-core' '$W/uv.log'"
chk "no fetch when the pin is present"   "! grep -q 'fetch' '$W/git.log'"
chk "playwright browser installed"       "grep -q 'install chromium' '$W/playwright.log'"
chk "miniwob clone pinned too"           "grep -q -- '-C $W/miniwob_stub checkout' '$W/git.log'"

########## part 2: pin_clone against real git ##########
eval "$(sed -n '/^pin_clone()/,/^}/p' "$REPO/setup.sh")"
newrepo() {  # $1 dir -> two commits, prints "old new"
  rm -rf "$1"; mkdir -p "$1"; git -C "$1" init -q
  git -C "$1" config user.email t@t; git -C "$1" config user.name t
  echo a > "$1/f"; git -C "$1" add f; git -C "$1" commit -qm a
  local old; old=$(git -C "$1" rev-parse HEAD)
  echo b > "$1/f"; git -C "$1" commit -qam b
  echo "$old $(git -C "$1" rev-parse HEAD)"
}
read -r OLD NEW <<< "$(newrepo "$W/src")"

# D: clean clone, pin present, origin unreachable -> no fetch needed, lands on pin
git clone -q "$W/src" "$W/c1"; git -C "$W/c1" remote set-url origin /nonexistent
out=$(pin_clone "$W/c1" unused "$OLD" 2>&1); rc=$?
chk "moves a clean clone onto the pin without fetching (rc=$rc)" \
    "[ $rc -eq 0 ] && [ \"\$(git -C '$W/c1' rev-parse HEAD)\" = '$OLD' ]"
chk "reports the move" "echo '$out' | grep -q '$OLD'"

# E: tracked modification -> refuse, HEAD unchanged
git clone -q "$W/src" "$W/c2"; echo dirty > "$W/c2/f"
out=$(pin_clone "$W/c2" unused "$OLD" 2>&1); rc=$?
chk "refuses a clone with uncommitted changes (rc=$rc)" \
    "[ $rc -eq 1 ] && [ \"\$(git -C '$W/c2' rev-parse HEAD)\" = '$NEW' ]"
chk "says why it refused" "echo '$out' | grep -q 'uncommitted changes'"

# F: untracked file only -> fine (the box clone carries one)
git clone -q "$W/src" "$W/c3"; touch "$W/c3/untracked"
( pin_clone "$W/c3" unused "$OLD" >/dev/null 2>&1 ); rc=$?
chk "untracked files do not block the checkout (rc=$rc)" \
    "[ $rc -eq 0 ] && [ -f '$W/c3/untracked' ]"

# G: missing dir -> clones, then pins
( pin_clone "$W/c4" "$W/src" "$OLD" >/dev/null 2>&1 ); rc=$?
chk "clones when the directory is missing (rc=$rc)" \
    "[ $rc -eq 0 ] && [ \"\$(git -C '$W/c4' rev-parse HEAD)\" = '$OLD' ]"

# H: pin not in the clone yet -> fetches it from origin
git clone -q "$W/src" "$W/c5" -b master 2>/dev/null || git clone -q "$W/src" "$W/c5"
git -C "$W/src" checkout -q -b extra; echo c > "$W/src/f"; git -C "$W/src" commit -qam c
LATER=$(git -C "$W/src" rev-parse HEAD)
( pin_clone "$W/c5" "$W/src" "$LATER" >/dev/null 2>&1 ); rc=$?
chk "fetches a pin the clone does not have yet (rc=$rc)" \
    "[ $rc -eq 0 ] && [ \"\$(git -C '$W/c5' rev-parse HEAD)\" = '$LATER' ]"

echo "----- $pass passed, $fail failed  ($W)"
[ $fail -eq 0 ]
