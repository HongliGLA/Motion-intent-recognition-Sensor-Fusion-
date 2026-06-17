import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(
            0,
            max_len,
            dtype=torch.float32
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class PureTransformerClassifier(nn.Module):
    """
    QK-only EMG/ACC block attention classifier.

    输入:
        x: [B, T, 64]

    重排后的数据顺序:
        前 48 列 ACC
        后 16 列 EMG

    模型方法保持不变:
        EMG -> Qe, Ke
        ACC -> Qa, Ka

        Q = concat(Qe, Qa)
        K = concat(Ke, Ka)

        S = QK^T / sqrt(d)

        S_ee = EMG-EMG
        S_ea = EMG-ACC
        S_ae = ACC-EMG
        S_aa = ACC-ACC

        pool / flatten -> classifier

    注意:
        没有 V
        没有 QK^T V
    """

    def __init__(self):
        super().__init__()

        assert cfg.D_MODEL % cfg.NHEAD == 0
        assert cfg.QK_FEATURE_MODE in ["pool", "flatten"]

        self.input_dim = cfg.INPUT_DIM
        self.num_classes = cfg.NUM_CLASSES
        self.window_size = cfg.WINDOW_SIZE
        self.d_model = cfg.D_MODEL
        self.nhead = cfg.NHEAD
        self.d_head = cfg.D_MODEL // cfg.NHEAD

        self.acc_channels = list(cfg.ACC_CHANNELS)
        self.emg_channels = list(cfg.EMG_CHANNELS)

        self.feature_mode = cfg.QK_FEATURE_MODE
        self.pool_size = cfg.QK_POOL_SIZE
        self.use_softmax = cfg.QK_USE_SOFTMAX

        all_channels = sorted(self.acc_channels + self.emg_channels)

        if all_channels != list(range(cfg.INPUT_DIM)):
            raise ValueError(
                f"ACC_CHANNELS 和 EMG_CHANNELS 必须刚好覆盖 0~{cfg.INPUT_DIM - 1} 所有通道"
            )

        print("Current model: QK-only Cross-Modal Block Attention")
        print("ACC channels:", self.acc_channels, "= 前 48 列")
        print("EMG channels:", self.emg_channels, "= 后 16 列")
        print("No V, no QK^T V")
        print("Feature mode:", self.feature_mode)

        self.emg_projection = nn.Linear(
            len(self.emg_channels),
            cfg.D_MODEL
        )

        self.acc_projection = nn.Linear(
            len(self.acc_channels),
            cfg.D_MODEL
        )

        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model=cfg.D_MODEL,
            max_len=cfg.WINDOW_SIZE + 10
        )

        self.emg_norm = nn.LayerNorm(cfg.D_MODEL)
        self.acc_norm = nn.LayerNorm(cfg.D_MODEL)

        self.qe = nn.Linear(cfg.D_MODEL, cfg.D_MODEL)
        self.ke = nn.Linear(cfg.D_MODEL, cfg.D_MODEL)

        self.qa = nn.Linear(cfg.D_MODEL, cfg.D_MODEL)
        self.ka = nn.Linear(cfg.D_MODEL, cfg.D_MODEL)

        self.dropout = nn.Dropout(cfg.DROPOUT)

        if self.feature_mode == "pool":
            classifier_in = (
                4
                * cfg.NHEAD
                * cfg.QK_POOL_SIZE
                * cfg.QK_POOL_SIZE
            )
        else:
            classifier_in = (
                4
                * cfg.NHEAD
                * cfg.WINDOW_SIZE
                * cfg.WINDOW_SIZE
            )

        self.feature_norm = nn.LayerNorm(classifier_in)

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, cfg.QK_CLASSIFIER_HIDDEN),
            nn.GELU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(cfg.QK_CLASSIFIER_HIDDEN, cfg.NUM_CLASSES)
        )

    def split_heads(self, x):
        """
        x: [B, T, D]
        return: [B, H, T, Dh]
        """

        B, T, D = x.shape

        x = x.view(
            B,
            T,
            self.nhead,
            self.d_head
        )

        x = x.transpose(
            1,
            2
        )

        return x

    def blocks_to_feature(
        self,
        S_ee,
        S_ea,
        S_ae,
        S_aa
    ):
        B = S_ee.size(0)

        if self.feature_mode == "flatten":
            feat = torch.cat(
                [
                    S_ee.reshape(B, -1),
                    S_ea.reshape(B, -1),
                    S_ae.reshape(B, -1),
                    S_aa.reshape(B, -1),
                ],
                dim=1
            )

            return feat

        P_ee = F.adaptive_avg_pool2d(
            S_ee,
            output_size=(self.pool_size, self.pool_size)
        )

        P_ea = F.adaptive_avg_pool2d(
            S_ea,
            output_size=(self.pool_size, self.pool_size)
        )

        P_ae = F.adaptive_avg_pool2d(
            S_ae,
            output_size=(self.pool_size, self.pool_size)
        )

        P_aa = F.adaptive_avg_pool2d(
            S_aa,
            output_size=(self.pool_size, self.pool_size)
        )

        feat = torch.cat(
            [
                P_ee.reshape(B, -1),
                P_ea.reshape(B, -1),
                P_ae.reshape(B, -1),
                P_aa.reshape(B, -1),
            ],
            dim=1
        )

        return feat

    def forward(self, x, return_blocks=False):
        """
        x: [B, T, C]
        """

        if x.size(1) != self.window_size:
            raise ValueError(
                f"输入时间长度 T={x.size(1)}，但 WINDOW_SIZE={self.window_size}"
            )

        if x.size(2) != self.input_dim:
            raise ValueError(
                f"输入通道数 C={x.size(2)}，但 INPUT_DIM={self.input_dim}"
            )

        emg_index = torch.as_tensor(
            self.emg_channels,
            dtype=torch.long,
            device=x.device
        )

        acc_index = torch.as_tensor(
            self.acc_channels,
            dtype=torch.long,
            device=x.device
        )

        xe = torch.index_select(
            x,
            dim=2,
            index=emg_index
        )

        xa = torch.index_select(
            x,
            dim=2,
            index=acc_index
        )

        he = self.emg_projection(xe)
        ha = self.acc_projection(xa)

        he = self.pos_encoding(he)
        ha = self.pos_encoding(ha)

        he = self.emg_norm(he)
        ha = self.acc_norm(ha)

        he = self.dropout(he)
        ha = self.dropout(ha)

        Qe = self.split_heads(self.qe(he))
        Ke = self.split_heads(self.ke(he))

        Qa = self.split_heads(self.qa(ha))
        Ka = self.split_heads(self.ka(ha))

        # 注意：Q/K 拼接顺序为 [EMG, ACC]，所以 S 的四块是 ee, ea, ae, aa。
        Q = torch.cat(
            [Qe, Qa],
            dim=2
        )

        K = torch.cat(
            [Ke, Ka],
            dim=2
        )

        S = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        S = S / math.sqrt(self.d_head)

        if self.use_softmax:
            S = torch.softmax(
                S,
                dim=-1
            )

        T = self.window_size

        S_ee = S[:, :, :T, :T]
        S_ea = S[:, :, :T, T:]
        S_ae = S[:, :, T:, :T]
        S_aa = S[:, :, T:, T:]

        feat = self.blocks_to_feature(
            S_ee,
            S_ea,
            S_ae,
            S_aa
        )

        feat = self.feature_norm(feat)

        logits = self.classifier(feat)

        if return_blocks:
            return logits, {
                "S": S,
                "S_ee": S_ee,
                "S_ea": S_ea,
                "S_ae": S_ae,
                "S_aa": S_aa,
            }

        return logits
