"""Staged-fusion ladder: same frozen experts, fusion mechanism as the only variable."""
G = "/gpfs/scratch/<account>/<user>/runs/raw_contrastive_90kcos_12k/gapfill"
src = open(f"{G}/p2fusion.slurm").read()
head, _, _ = src.partition("#run_cell")
head = head.replace("--job-name=p2fusion", "--job-name=p2staged")

# run_cell gains: $4 = cross_modal (m0 concat | m1 cross-attention), $5 = experts
head = head.replace('  NAME=$1; MOD=$2; POE=$3; PROBE=${4:-}',
                    '  NAME=$1; MOD=$2; POE=$3; ARCH=$4; EXPERTS=$5')
head = head.replace(
    '''printf '{"cells":[{"cross_modal":"m1",''',
    '''printf '{"cells":[{"cross_modal":"'"$ARCH"'",''')
head = head.replace('  [ -n "$PROBE" ] && EXTRA="$EXTRA --extra-visible-bits $PROBE"',
                    '  [ "$ARCH" = "m1" ] && EXTRA="$EXTRA --freeze-experts"')
# Only a readout (or the fusion block) trains, so the contrastive objective has
# no encoder to reshape; switch it off rather than let it train a side branch.
head = head.replace(
    "--mlc-loss bce --mask-invisible-bits $EXTRA",
    "--mlc-loss bce --mask-invisible-bits --cross-modal-beta 0 \\\n"
    "        --expert-checkpoints \"$EXPERTS\" $EXTRA")

V = f"{G}/p2V_base/runs/m1_i3_p2_V.pt"
A = f"{G}/p2A_base/runs/m1_i3_p2_A.pt"
T = f"{G}/p2T_base/runs/m1_i3_p2_T.pt"
E = {"VA": f"v={V},a={A}", "AT": f"a={A},t={T}",
     "VT": f"v={V},t={T}", "VAT": f"v={V},a={A},t={T}"}

lines = []
for mod in ("AT", "VA", "VT", "VAT"):          # AT first: it carries text_shuffle
    lines.append(f'run_cell p2s{mod}  {mod} 0 m0 "{E[mod]}"')
    lines.append(f'run_cell p2fz{mod} {mod} 0 m1 "{E[mod]}"')
lines.append('echo "===== P2STAGED_ALL_DONE $(date) ====="')
open(f"{G}/p2staged.slurm", "w").write(head + "\n".join(lines) + "\n")
print("written")
