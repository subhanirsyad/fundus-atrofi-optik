"""Inferensi: citra fundus -> FOV + CLAHE -> Swin-UNet (optic disc) -> ROI -> ResNet-18 (3 kelas) + Grad-CAM.

Semua fungsi disalin dari notebook Zeppelin:
- swinunet_segmentasi_latih_62_tambahan.zpln (arsitektur Swin-UNet)
- swinunet_resnet18_5kasus_margin010_79.zpln (pra-proses, ROI, Klasifier)
- swinunet_resnet18_gradcam_fold2.zpln (Grad-CAM)
Yang diganti hanya pembacaan citra: yavai.read_image -> cv2.imdecode dari file yang diunggah.
"""
import base64
import os
import threading
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# ---------- konfigurasi (sama dengan notebook 79) ----------
ROI_SIZE = 224
FOV_THRESH = 10
FOV_MARGIN = 0.02
SEG_THRESHOLD = 0.5
ROI_MARGIN = 0.10
DEPTHS = (2, 2, 2, 2)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NAMA_KELAS = ["Normal", "Total", "Parsial"]      # urutan indeks keluaran model (KASUS["3 Kelas"])

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
SWIN_FILE = "swin_unet_imagenet_disc_label_dokter_adamw_2tahap_latih_tambahan.pth"
RESNET_FILE = "resnet18_roi_3_kelas_terbaik.pth"   # dari notebook final_lengkap (fold terbaik 3 Kelas)
device = "cpu"
torch.set_num_threads(min(8, os.cpu_count() or 4))


# ======================================================================
# Arsitektur Swin-UNet (verbatim dari notebook segmentasi)
# ======================================================================
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


def patch_expand_rearrange(x, H, W, C, p):
    B = x.shape[0]
    c_out = C // (p * p)
    x = x.view(B, H, W, p, p, c_out).permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(B, H * p, W * p, c_out)


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.dim, self.window_size, self.num_heads = dim, window_size, num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords = torch.stack(torch.meshgrid(
            [torch.arange(window_size[0]), torch.arange(window_size[1])], indexing="ij"))
        cf = torch.flatten(coords, 1)
        rc = (cf[:, :, None] - cf[:, None, :]).permute(1, 2, 0).contiguous()
        rc[:, :, 0] += window_size[0] - 1
        rc[:, :, 1] += window_size[1] - 1
        rc[:, :, 0] *= 2 * window_size[1] - 1
        self.register_buffer("relative_position_index", rc.sum(-1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        rb = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        rb = rb.view(self.window_size[0] * self.window_size[1],
                     self.window_size[0] * self.window_size[1], -1).permute(2, 0, 1).contiguous()
        attn = attn + rb.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        x = (self.softmax(attn) @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0):
        super().__init__()
        self.dim, self.input_resolution = dim, input_resolution
        self.num_heads, self.window_size, self.shift_size = num_heads, window_size, shift_size
        if min(input_resolution) <= window_size:
            self.shift_size, self.window_size = 0, min(input_resolution)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, (self.window_size, self.window_size), num_heads)
        self.norm2 = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))
        if self.shift_size > 0:
            H, W = input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            cnt = 0
            for hs in (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None)):
                for ws in (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size),
                           slice(-self.shift_size, None)):
                    img_mask[:, hs, ws, :] = cnt
                    cnt += 1
            mw = window_partition(img_mask, self.window_size).view(-1, self.window_size ** 2)
            am = mw.unsqueeze(1) - mw.unsqueeze(2)
            am = am.masked_fill(am != 0, float(-100.0)).masked_fill(am == 0, float(0.0))
        else:
            am = None
        self.register_buffer("attn_mask", am)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        xw = window_partition(x, self.window_size).view(-1, self.window_size ** 2, C)
        aw = self.attn(xw, mask=self.attn_mask)
        x = window_reverse(aw.view(-1, self.window_size, self.window_size, C), self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = shortcut + x.view(B, H * W, C)
        return x + self.mlp(self.norm2(x))


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x = torch.cat([x[:, 0::2, 0::2, :], x[:, 1::2, 0::2, :],
                       x[:, 0::2, 1::2, :], x[:, 1::2, 1::2, :]], -1).view(B, -1, 4 * C)
        return self.reduction(self.norm(x))


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2):
        super().__init__()
        self.input_resolution = input_resolution
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        x = patch_expand_rearrange(x.view(B, H, W, C), H, W, C, p=2)
        return self.norm(x.view(B, -1, C // 4))


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4):
        super().__init__()
        self.input_resolution, self.dim_scale, self.output_dim = input_resolution, dim_scale, dim
        self.expand = nn.Linear(dim, (dim_scale ** 2) * dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        x = patch_expand_rearrange(x.view(B, H, W, C), H, W, C, p=self.dim_scale)
        return self.norm(x.view(B, -1, self.output_dim))


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4.0, downsample=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim, input_resolution, num_heads, window_size,
                                 0 if (i % 2 == 0) else window_size // 2, mlp_ratio)
            for i in range(depth)])
        self.downsample = downsample(input_resolution, dim=dim) if downsample is not None else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        skip = x
        if self.downsample is not None:
            x = self.downsample(x)
        return x, skip


class BasicLayerUp(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4.0, upsample=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim, input_resolution, num_heads, window_size,
                                 0 if (i % 2 == 0) else window_size // 2, mlp_ratio)
            for i in range(depth)])
        self.upsample = upsample(input_resolution, dim=dim) if upsample is not None else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class SwinUnet(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1, embed_dim=96,
                 depths=(2, 2, 2, 2), depths_decoder=(2, 2, 2, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=4.0):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim, self.patch_size = embed_dim, patch_size
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm_embed = nn.LayerNorm(embed_dim)
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            dim = embed_dim * (2 ** i)
            res = (self.patches_resolution[0] // (2 ** i), self.patches_resolution[1] // (2 ** i))
            self.layers.append(BasicLayer(dim, res, depths[i], num_heads[i], window_size, mlp_ratio,
                                          PatchMerging if i < self.num_layers - 1 else None))
        self.norm_bottleneck = nn.LayerNorm(embed_dim * (2 ** (self.num_layers - 1)))
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i in range(self.num_layers):
            j = self.num_layers - 1 - i
            dim = embed_dim * (2 ** j)
            res = (self.patches_resolution[0] // (2 ** j), self.patches_resolution[1] // (2 ** j))
            if i == 0:
                self.layers_up.append(PatchExpand(res, dim=dim))
                self.concat_back_dim.append(nn.Identity())
            else:
                up = PatchExpand if i < self.num_layers - 1 else None
                self.layers_up.append(BasicLayerUp(dim, res, depths_decoder[j], num_heads[j],
                                                   window_size, mlp_ratio, up))
                self.concat_back_dim.append(nn.Linear(2 * dim, dim))
        self.norm_up = nn.LayerNorm(embed_dim)
        self.up_final = FinalPatchExpand_X4(self.patches_resolution, dim=embed_dim, dim_scale=patch_size)
        self.output = nn.Conv2d(embed_dim, num_classes, kernel_size=1, bias=False)

    def forward(self, x):
        B = x.shape[0]
        x = self.norm_embed(self.patch_embed(x).flatten(2).transpose(1, 2))
        skips = []
        for layer in self.layers:
            x, skip = layer(x)
            skips.append(skip)
        x = self.norm_bottleneck(x)
        for i, layer_up in enumerate(self.layers_up):
            if i == 0:
                x = layer_up(x)
            else:
                x = self.concat_back_dim[i](torch.cat([x, skips[self.num_layers - 1 - i]], -1))
                x = layer_up(x)
        x = self.up_final(self.norm_up(x))
        Hf = self.patches_resolution[0] * self.patch_size
        Wf = self.patches_resolution[1] * self.patch_size
        return self.output(x.view(B, Hf, Wf, self.embed_dim).permute(0, 3, 1, 2))


# ======================================================================
# Klasifier ResNet-18 (notebook 79). weights=None: bobot diisi dari file .pth
# ======================================================================
class Klasifier(nn.Module):
    def __init__(self, n_kelas=3):
        super().__init__()
        self.backbone = torchvision.models.resnet18(weights=None)
        d = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.n_view = 1
        self.head = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, n_kelas))

    def forward(self, x):
        return self.head(self.backbone(x))


# ======================================================================
# Pra-proses dan ROI (notebook 79, paragraf 13)
# ======================================================================
def get_fov_bbox(img, thresh=FOV_THRESH):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = gray.shape[:2]
        return 0, 0, w, h
    return cv2.boundingRect(max(contours, key=cv2.contourArea))


def crop_to_fov(img, thresh=FOV_THRESH, margin=FOV_MARGIN):
    x, y, w, h = get_fov_bbox(img, thresh=thresh)
    mx, my = int(w * margin), int(h * margin)
    H, W = img.shape[:2]
    return img[max(0, y - my):min(H, y + h + my), max(0, x - mx):min(W, x + w + mx)]


def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def masker_fov(img, kikis=0.10):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (g > FOV_THRESH).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        mask = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    r = np.sqrt(max(mask.sum(), 1) / np.pi)
    k = max(3, int(2 * kikis * r) | 1)
    return cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def ke_tensor_rgb(rgb):
    x = torch.from_numpy(np.ascontiguousarray(rgb)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return (x - torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)) / torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)


@torch.no_grad()
def segmentasi_disc(model, img_fov_clahe, img_size=ROI_SIZE):
    rgb = cv2.cvtColor(cv2.resize(img_fov_clahe, (img_size, img_size), interpolation=cv2.INTER_LINEAR),
                       cv2.COLOR_BGR2RGB)
    return torch.sigmoid(model(ke_tensor_rgb(rgb).to(device)))[0, 0].cpu().numpy()


def lokalisasi_klasik(img):
    H, W = img.shape[:2]
    sisi = min(H, W)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = max(3, int(0.02 * sisi) | 1)
    closed = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    rk = max(6, int(0.09 * sisi))
    resp = np.where(masker_fov(img) > 0, cv2.blur(closed, (2 * rk + 1, 2 * rk + 1)), -1.0)
    cy, cx = np.unravel_index(int(np.argmax(resp)), resp.shape)
    return float(cx - rk), float(cy - rk), float(cx + rk), float(cy + rk)


def ekstrak_roi(img_sumber, prob, img_fov_clahe, thr=SEG_THRESHOLD, margin=ROI_MARGIN, out_size=ROI_SIZE):
    H, W = img_sumber.shape[:2]
    fov224 = masker_fov(cv2.resize(img_fov_clahe, (prob.shape[1], prob.shape[0]), interpolation=cv2.INTER_AREA))
    mask = ((prob > thr) & (fov224 > 0)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        j = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        bx, by, bw, bh = (int(v) for v in stats[j, :4])
        sx, sy = W / prob.shape[1], H / prob.shape[0]
        x0, y0, x1, y1 = bx * sx, by * sy, (bx + bw) * sx, (by + bh) * sy
        sumber = "swin"
        mask_disc = (lab == j).astype(np.uint8)
    else:
        x0, y0, x1, y1 = lokalisasi_klasik(img_fov_clahe)
        sumber = "cadangan"
        mask_disc = np.zeros_like(mask)
    mx, my = int(margin * (x1 - x0)), int(margin * (y1 - y0))
    x0, x1, y0, y1 = x0 - mx, x1 + mx, y0 - my, y1 + my
    sisi = int(round(max(x1 - x0, y1 - y0)))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    X0, Y0 = int(round(cx - sisi / 2)), int(round(cy - sisi / 2))
    X1, Y1 = X0 + sisi, Y0 + sisi
    pad = max(0, -X0, -Y0, X1 - W, Y1 - H)
    src = cv2.copyMakeBorder(img_sumber, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0) if pad else img_sumber
    roi = src[Y0 + pad:Y1 + pad, X0 + pad:X1 + pad]
    interp = cv2.INTER_AREA if sisi > out_size else cv2.INTER_CUBIC
    roi = cv2.resize(roi, (out_size, out_size), interpolation=interp)
    return roi, {"sumber": sumber, "kotak": (X0, Y0, X1, Y1), "mask_disc": mask_disc,
                 "rasio_luas_disc": float(mask_disc.sum()) / mask_disc.size}


# ======================================================================
# Grad-CAM layer4 (notebook gradcam fold 2)
# ======================================================================
def gradcam(model, x):
    simpan = {}

    def kait(m, i, o):
        simpan["a"] = o
        o.register_hook(lambda g: simpan.__setitem__("g", g))

    h = model.backbone.layer4.register_forward_hook(kait)
    model.eval()
    x = x.clone().requires_grad_(True)
    logit = model(x)
    model.zero_grad()
    logit[torch.arange(len(x)), logit.argmax(1)].sum().backward()
    h.remove()
    w = simpan["g"].mean((2, 3), keepdim=True)
    cam = F.relu((w * simpan["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    cam = cam / cam.flatten(1).max(1)[0].clamp_min(1e-8)[:, None, None]
    return cam.detach().cpu().numpy(), torch.softmax(logit.detach(), 1)[0].cpu().numpy()


# ======================================================================
# Pemuatan model dan satu panggilan analisis
# ======================================================================
SWIN = RESNET = None
GALAT_MUAT = None
_kunci = threading.Lock()   # ponytail: satu analisis sekaligus (hook Grad-CAM dipasang di model bersama); cukup untuk demo sidang


def muat_model():
    global SWIN, RESNET, GALAT_MUAT
    hilang = [f for f in (SWIN_FILE, RESNET_FILE) if not os.path.exists(os.path.join(MODEL_DIR, f))]
    if hilang:
        GALAT_MUAT = {"folder": MODEL_DIR, "hilang": hilang}
        return
    swin = SwinUnet(img_size=ROI_SIZE, patch_size=4, in_chans=3, num_classes=1, embed_dim=96,
                    depths=DEPTHS, depths_decoder=DEPTHS, num_heads=(3, 6, 12, 24), window_size=7)
    swin.load_state_dict(torch.load(os.path.join(MODEL_DIR, SWIN_FILE), map_location="cpu"), strict=True)
    resnet = Klasifier(len(NAMA_KELAS))
    resnet.load_state_dict(torch.load(os.path.join(MODEL_DIR, RESNET_FILE), map_location="cpu"), strict=True)
    for m in (swin, resnet):
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
    SWIN, RESNET, GALAT_MUAT = swin, resnet, None


def _jpeg(rgb, q=90):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def analisis(data: bytes):
    """Bytes file citra -> dict hasil. ValueError kalau file bukan citra."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("File tidak bisa dibaca sebagai citra.")
    t0 = time.perf_counter()
    with _kunci:
        fov = crop_to_fov(img)
        fov_clahe = apply_clahe(fov)
        prob = segmentasi_disc(SWIN, fov_clahe)
        roi_bgr, info = ekstrak_roi(fov, prob, fov_clahe)      # CLAHE_UNTUK_RESNET = False
        roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        cam, p = gradcam(RESNET, ke_tensor_rgb(roi))
    detik = time.perf_counter() - t0

    # Panel segmentasi: FOV + CLAHE 224 (masukan Swin-UNet), kontur disc dan kotak ROI
    seg = cv2.cvtColor(cv2.resize(fov_clahe, (ROI_SIZE, ROI_SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    seg = cv2.resize(seg, (448, 448), interpolation=cv2.INTER_CUBIC)
    kontur, _ = cv2.findContours(cv2.resize(info["mask_disc"], (448, 448), interpolation=cv2.INTER_NEAREST),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(seg, kontur, -1, (0, 200, 255), 2)
    H, W = fov.shape[:2]
    X0, Y0, X1, Y1 = info["kotak"]
    cv2.rectangle(seg, (int(X0 * 448 / W), int(Y0 * 448 / H)), (int(X1 * 448 / W), int(Y1 * 448 / H)),
                  (255, 255, 255), 1)
    panas = cv2.cvtColor(cv2.applyColorMap((cam[0] * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)

    return {
        "kelas": NAMA_KELAS[int(p.argmax())],
        "probabilitas": {k: float(v) for k, v in zip(NAMA_KELAS, p)},
        "sumber_roi": info["sumber"],
        "rasio_luas_disc": info["rasio_luas_disc"],
        "ukuran_asli": [int(img.shape[1]), int(img.shape[0])],
        "detik": round(detik, 2),
        "gambar": {"segmentasi": _jpeg(seg), "roi": _jpeg(roi), "gradcam": _jpeg(panas)},
    }
