"""Oracle-PoE building blocks: observable-class codes, the log-odds table,
and the tug-of-war equilibrium the loss is designed around."""
import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from rmfdn_ablation.trainer import _oracle_codes, build_oracle_table
from rmfdn_ablation.config import MANIPULATION_TECHNIQUES

V_VISIBLE = (0, 1)  # lip_sync, face_swap
TTS = MANIPULATION_TECHNIQUES.index("tts")
AUD_SH = MANIPULATION_TECHNIQUES.index("audio_shuffle")


def _clip(bits, is_fake=1):
    return SimpleNamespace(technique_targets=list(bits), is_fake=is_fake)


PRISTINE = _clip([0, 0, 0, 0, 0, 0, 0], is_fake=0)
FACE = _clip([1, 1, 0, 0, 1, 0, 1])          # face packages also set lip_sync
LIP = _clip([1, 0, 1, 0, 0, 1, 0])
NONE_TTS = _clip([0, 0, 1, 0, 0, 1, 0])      # visually clean, tts present
NONE_RVC = _clip([0, 0, 0, 1, 0, 0, 1])      # visually clean, tts absent


def _tensors(clips):
    targets = torch.tensor([c.technique_targets for c in clips], dtype=torch.float32)
    is_fake = torch.tensor([c.is_fake for c in clips])
    return targets, is_fake


def test_codes_reproduce_visual_class_partition():
    # V hucresi icin kod = is_fake + 2*lip + 4*face -> PRISTINE/NONE/LIP/FACE
    targets, is_fake = _tensors([PRISTINE, NONE_TTS, LIP, FACE])
    assert _oracle_codes(targets, is_fake, V_VISIBLE).tolist() == [0, 1, 3, 7]


def test_table_matches_hand_computed_frequencies():
    # NONE sinifi: 3 tts'li + 1 tts'siz -> p(tts|NONE)=0.75 -> log(3)
    clips = [NONE_TTS, NONE_TTS, NONE_TTS, NONE_RVC, PRISTINE, FACE, LIP]
    table = build_oracle_table(clips, V_VISIBLE)
    assert table.shape == (8, len(MANIPULATION_TECHNIQUES))
    assert math.isclose(table[1, TTS].item(), math.log(3), rel_tol=1e-5)


def test_structural_zeros_clamped_and_visible_columns_zeroed():
    clips = [NONE_TTS, NONE_RVC, PRISTINE, FACE, LIP]
    table = build_oracle_table(clips, V_VISIBLE, eps=1e-3)
    clamped = math.log(1e-3 / (1 - 1e-3))
    assert math.isclose(table[1, AUD_SH].item(), clamped, rel_tol=1e-4)
    assert math.isclose(table[0, TTS].item(), clamped, rel_tol=1e-4)
    assert torch.isfinite(table).all()
    assert (table[:, list(V_VISIBLE)] == 0).all()          # gorunur sutunlar duz BCE
    unseen = [c for c in range(8) if c not in (0, 1, 3, 7)]
    assert (table[unseen] == 0).all()                       # gorulmemis desen = ofset yok


def test_poe_equilibrium_kills_class_statistic_gradient():
    # 79 tts'li + 21 tts'siz NONE klibi; model hepsine ayni e skorunu basabilir.
    clips = [NONE_TTS] * 79 + [NONE_RVC] * 21
    targets, is_fake = _tensors(clips)
    table = build_oracle_table(clips, V_VISIBLE)
    offset = table[_oracle_codes(targets, is_fake, V_VISIBLE)][:, TTS]
    y = targets[:, TTS]

    e = torch.zeros(1, requires_grad=True)
    F.binary_cross_entropy_with_logits(e.expand(100) + offset, y).backward()
    poe_grad = abs(e.grad.item())

    e2 = torch.zeros(1, requires_grad=True)
    F.binary_cross_entropy_with_logits(e2.expand(100), y).backward()
    plain_grad = abs(e2.grad.item())

    assert poe_grad < 1e-5          # oracle dengeyi kurdu: net itki yok
    assert plain_grad > 0.2         # duz BCE'de hasat itkisi buyuk (|0.5-0.79|)


def test_visible_loo_table_kills_face_implies_lip():
    from rmfdn_ablation.trainer import build_visible_loo_table
    LIP_B, FACE_B = 0, 1
    # FACE paketleri lip'i de set eder; LIP paketleri face'siz; NONE ikisi de 0.
    clips = [FACE] * 40 + [LIP] * 30 + [NONE_TTS] * 20 + [PRISTINE] * 10
    table = build_visible_loo_table(clips, V_VISIBLE)
    codes = _oracle_codes(*_tensors(clips), V_VISIBLE)
    c_face, c_lip, c_none = codes[0].item(), codes[40].item(), codes[70].item()

    # face=1 iken lip HER ZAMAN 1 -> oracle kesin evet der, ogrenilecek sey yok
    assert table[c_face, LIP_B] > 6.0
    # lip=0 iken face HER ZAMAN 0 -> oracle kesin hayir
    assert table[c_none, FACE_B] < -6.0
    # lip=1 iken face belirsiz (40 face / 70 lipli) -> orta deger, kanit yasar
    assert -1.0 < table[c_lip, FACE_B] < 1.0
    # gorunmez sutunlara dokunulmaz (maske onlari zaten kesiyor)
    invisible = [i for i in range(7) if i not in V_VISIBLE]
    assert (table[:, invisible] == 0).all()


def test_aux_head_restores_trunk_gradient_where_oracle_saturates():
    """The measured head's loss is unchanged; the aux head re-supplies the
    gradient that saturated oracle regions withhold (e.g. genuine clips)."""
    import torch
    import torch.nn.functional as F
    from rmfdn_ablation.trainer import _compute_training_loss

    torch.manual_seed(0)
    feature = torch.randn(4, 8, requires_grad=True)
    head = torch.nn.Linear(8, 7)
    aux = torch.nn.Linear(8, 7)
    targets = torch.zeros(4, 7)                       # genuine clips: all bits 0
    saturated = torch.full((4, 7), -6.9)              # oracle: already explained
    mask = torch.zeros(7); mask[:2] = 1.0             # visible bits lip, face

    def loss_for(with_aux):
        outputs = {"technique_logits": head(feature) + saturated,
                   "binary_logits": torch.zeros(4, 2)}
        if with_aux:
            outputs["aux_technique_logits"] = aux(feature)
        per_bit = F.binary_cross_entropy_with_logits(
            outputs["technique_logits"], targets, reduction="none")
        loss = (per_bit * mask).sum() / (mask.sum() * 4)
        if "aux_technique_logits" in outputs:
            aux_per_bit = F.binary_cross_entropy_with_logits(
                outputs["aux_technique_logits"], targets, reduction="none")
            loss = loss + (aux_per_bit * mask).sum() / (mask.sum() * 4)
        return loss

    grad_without = torch.autograd.grad(loss_for(False), feature, retain_graph=False)[0]
    grad_with = torch.autograd.grad(loss_for(True), feature)[0]
    assert grad_without.abs().mean() < 1e-3           # oracle killed the signal
    assert grad_with.abs().mean() > 10 * grad_without.abs().mean()


def test_aux_head_is_optional_and_never_scored():
    from unittest import mock
    import torch
    from rmfdn_ablation import RMFDN as rmfdn_module

    class _Tiny(torch.nn.Module):
        def __init__(self, *_a):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
        def forward(self, values, *_a):
            return values.float().mean(1)

    config = {"embed_dim": 2, "attention_heads": 1, "contrastive_dim": 2,
              "reference_alpha": 0.001}
    batch = {"frames": torch.randn(2, 3, 2)}
    with mock.patch.object(rmfdn_module, "VisualEncoder", _Tiny):
        plain = rmfdn_module.RMFDN(config, ["v"])
        withaux = rmfdn_module.RMFDN({**config, "aux_plain_head": True}, ["v"])
    assert "aux_technique_logits" not in plain(batch)
    out = withaux(batch)
    assert out["aux_technique_logits"].shape == (2, 7)
    # the measured head is a different module: aux never contaminates it
    assert not torch.equal(out["aux_technique_logits"], out["technique_logits"])


def test_visible_loo_table_no_fake_cond_marginalises_real_fake():
    import math
    from types import SimpleNamespace
    import torch
    from rmfdn_ablation.trainer import build_visible_loo_table

    def clip(bits, fake):
        return SimpleNamespace(technique_targets=list(bits), is_fake=fake)

    # visible bits (0, 1); bit0 rate differs sharply between real and fake
    # within the same sibling pattern (bit1 = 0).
    clips = ([clip([1, 0, 0, 0, 0, 0, 0], 1)] * 3 +   # fake, sib=0, bit0=1
             [clip([0, 0, 0, 0, 0, 0, 0], 1)] * 1 +   # fake, sib=0, bit0=0
             [clip([0, 0, 0, 0, 0, 0, 0], 0)] * 4)    # real, sib=0, bit0=0

    with_fake = build_visible_loo_table(clips, (0, 1), fake_cond=True)
    no_fake = build_visible_loo_table(clips, (0, 1), fake_cond=False)

    # code = is_fake + 2*bit0 + 4*bit1 -> sib(bit1)=0 rows: codes 0 (real) / 1 (fake)
    # fake_cond=True separates them...
    assert with_fake[1, 0] > with_fake[0, 0]
    # ...fake_cond=False must give real/fake row pairs identical entries,
    assert torch.equal(no_fake[0], no_fake[1])
    # equal to the marginalised conditional P(bit0 | sib=0) = 3/8.
    expected = math.log((3 / 8) / (1 - 3 / 8))
    assert abs(no_fake[0, 0].item() - expected) < 1e-5


def test_frozen_offset_table_places_rows_at_fakeless_codes(tmp_path):
    from rmfdn_ablation.trainer import build_frozen_offset_table

    header = ("cell,variant,bit,role,bias_source,oracle_key,"
              "conditioning_pattern,logodds_F0,logodds_F1,logodds_F2,logodds_F3\n")
    rows = [
        "# comment line\n",
        header,
        "V,v4,lip,observable,sibling_oracle,sibling|V|lip|0,0,-0.5,-0.6,-0.7,-0.8\n",
        "V,v4,lip,observable,sibling_oracle,sibling|V|lip|1,1,6.9068,6.9068,6.9068,6.9068\n",
        "V,v4,face,observable,sibling_oracle,sibling|V|face|0,0,-6.9068,-6.9068,-6.9068,-6.9068\n",
        "V,v4,face,observable,sibling_oracle,sibling|V|face|1,1,-0.12,-0.11,-0.13,-0.17\n",
        "V,v2,lip,observable,sibling_oracle,sibling|V|lip|0,0,-0.5,-0.6,-0.7,-0.8\n",
        "V,v2,tts,free,class_oracle,class|V|1|0|tts,1|0,-0.677,-0.714,-0.663,-0.690\n",
        "AT,v4,tts,observable,sibling_oracle,sibling|AT|tts|0|0,0|0,1.5,1.5,1.5,1.5\n",
    ]
    path = tmp_path / "run_matrix.csv"
    path.write_text("".join(rows))

    table = build_frozen_offset_table(str(path), "V", "v4", "F0", (0, 1))
    # runtime codes are built with is_fake zeroed -> only even rows may carry
    # offsets, and the sibling offset must not depend on the bit's own value
    assert table.shape == (8, len(MANIPULATION_TECHNIQUES))
    assert torch.all(table[1::2] == 0)
    # lip's offset depends on face only: rows (lip=0,face=0) and (lip=1,face=0)
    assert table[0, 0] == table[2, 0] == -0.5
    assert table[4, 0] == table[6, 0] == 6.9068
    # face's offset depends on lip only
    assert table[0, 1] == table[4, 1] == -6.9068
    assert table[2, 1] == table[6, 1] == -0.12
    # fold column selection
    assert build_frozen_offset_table(str(path), "V", "v4", "F3", (0, 1))[0, 0] == -0.8

    # v2 adds the class-oracle offset for a free bit at the full observable
    # pattern (lip=1, face=0), and only there
    t2 = build_frozen_offset_table(str(path), "V", "v2", "F0", (0, 1))
    assert t2[2, TTS] == -0.677
    assert t2[0, TTS] == t2[4, TTS] == t2[6, TTS] == 0.0

    # two-bit conditioning pattern maps position-wise onto the observable set
    t3 = build_frozen_offset_table(str(path), "AT", "v4", "F0", (2, 3, 5))
    # pattern 0|0 conditions on (rvc, llm); it must cover both tts values
    codes = _oracle_codes(torch.tensor([[0., 0., 0., 0., 0., 0., 0.],
                                        [0., 0., 1., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 1., 0.]]),
                          torch.tensor([0, 0, 0]), (2, 3, 5))
    assert t3[codes[0], TTS] == t3[codes[1], TTS] == 1.5
    assert t3[codes[2], TTS] == 0.0


def test_effective_visible_indices_override_pins_the_set():
    from rmfdn_ablation.config import effective_visible_indices

    # built-in map: audio+text sees tsh; the Phase-1 override removes it
    assert 6 in effective_visible_indices(["a", "t"], {})
    pinned = effective_visible_indices(
        ["a", "t"], {"override_visible_bits": ["tts", "rvc", "llm"]})
    assert pinned == (2, 3, 5)


def test_no_binary_head_drops_the_detection_branch():
    from unittest import mock
    from rmfdn_ablation import RMFDN as rmfdn_module
    from rmfdn_ablation.trainer import _compute_training_loss

    class _Tiny(torch.nn.Module):
        def __init__(self, *_a):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
        def forward(self, values, *_a):
            return values.float().mean(1)

    config = {"embed_dim": 2, "attention_heads": 1, "contrastive_dim": 2,
              "reference_alpha": 0.001, "no_binary_head": True}
    batch = {"frames": torch.randn(2, 3, 2)}
    with mock.patch.object(rmfdn_module, "VisualEncoder", _Tiny):
        model = rmfdn_module.RMFDN(config, ["v"])
    outputs = model(batch)
    assert "binary_logits" not in outputs
    assert not any("binary" in k for k in model.state_dict())

    # the loss composes without a detection term (and without the identity
    # contrastive when its weight is zero)
    targets = torch.tensor([[0.] * 7, [1.] + [0.] * 6])
    loss_batch = {"frames": batch["frames"], "is_fake": torch.tensor([0, 1]),
                  "identity": ["a", "b"]}
    loss = _compute_training_loss(outputs, loss_batch, targets, ["v"], 0.1,
                                  cross_modal_beta=0.0, identity_gamma=0.0,
                                  reference_outputs=None, accum_steps=1)
    assert torch.isfinite(loss)
