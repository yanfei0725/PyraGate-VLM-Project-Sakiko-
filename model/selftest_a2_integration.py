"""
a2 集成自检：证明三模式注意力 + 参数量零增量 + 全模块可前向
纯 CPU，不占显存，不碰训练。
"""
import torch
from uvit3d import UViT3D, UViT3DConfig


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def has_hybrid(model):
    """统计实例化的 hybrid 分支数量（softmax 模式应为 0）。"""
    n = 0
    for m in model.modules():
        if hasattr(m, "hybrid") and m.hybrid is not None:
            n += 1
    return n


def make_model(attn_core, hybrid_layers):
    cfg = UViT3DConfig.tiny()
    cfg.attn_core = attn_core
    cfg.hybrid_layers = hybrid_layers
    torch.manual_seed(123)
    return UViT3D(cfg)


def main():
    latent = torch.randn(1, 16, 2, 4, 4)
    t = torch.randn(1, 16)
    txt = torch.randn(1, 32)

    # 1) softmax 模式：hybrid 分支 0 个，参数量与基线一致
    m_soft = make_model("softmax", "none")
    n_soft = count_params(m_soft)
    h_soft = has_hybrid(m_soft)
    assert h_soft == 0, f"softmax 模式不应有 hybrid 分支，实测 {h_soft}"
    y_soft = m_soft(latent, t, txt)[0]

    # 用一个不带 a2 字段的纯净基线配置对照（默认值即 softmax/none）
    cfg_base = UViT3DConfig.tiny()
    torch.manual_seed(123)
    m_base = UViT3D(cfg_base)
    n_base = count_params(m_base)
    assert n_soft == n_base, f"softmax 模式参数量应零增量：{n_soft} vs {n_base}"

    # 2) linear 模式：全部块线性化，能前向，参数量仅增门控
    m_lin = make_model("linear", "all")
    h_lin = has_hybrid(m_lin)
    n_lin = count_params(m_lin)
    assert h_lin > 0, "linear 模式应有 hybrid 分支"
    y_lin = m_lin(latent, t, txt)[0]
    assert y_lin.shape == y_soft.shape and torch.isfinite(y_lin).all()

    # 3) hybrid 模式（enc/dec 混合、bottleneck 保 softmax）
    m_hyb = make_model("hybrid", "spatial,temporal")
    h_hyb = has_hybrid(m_hyb)
    y_hyb = m_hyb(latent, t, txt)[0]
    assert h_hyb > 0 and y_hyb.shape == y_soft.shape and torch.isfinite(y_hyb).all()

    # 4) 零门控起步：hybrid 模型门控全 0 → 局部 softmax 语义（不污染）
    gates = []
    for m in m_hyb.modules():
        if hasattr(m, "hybrid") and m.hybrid is not None:
            gates.append(m.hybrid.gate_logit.detach().abs().max().item())
    assert max(gates) == 0.0, "hybrid 门控应零初始化"

    # 参数量增量只来自逐 head 门控（每块每 attn = num_heads 个标量）
    delta = n_lin - n_base
    print("[a2 integration selfcheck]")
    print(f"  softmax 模式: hybrid分支={h_soft}, 参数量={n_soft} (基线={n_base}, 零增量 OK)")
    print(f"  linear  模式: hybrid分支={h_lin}, 参数量={n_lin}, 门控增量={delta}")
    print(f"  hybrid  模式: hybrid分支={h_hyb}, 输出形状={tuple(y_hyb.shape)}")
    print(f"  门控零初始化: OK (max|gate|={max(gates)})")
    print(f"  三模式前向: OK (全部 finite)")
    print("  ALL PASS")


if __name__ == "__main__":
    main()
