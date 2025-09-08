# src/models/simplebev_xs.py
import argparse, os, time, json, math, random, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageEnhance
from tqdm import tqdm

# optional OpenCV for blur/dilate and colormaps in viz/rasterizer
try:
    import cv2
    _HAVE_CV2 = True
except Exception:
    _HAVE_CV2 = False


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def _norm01(x):
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    return np.clip(x, 0.0, 1.0)

def _heat_to_img(x, p_low=2.0, p_high=98.0, gamma=1.0, use_colormap=False):
    """
    Convert a single-channel map to RGB with percentile contrast stretching,
    optional gamma correction, and optional perceptual colormap.
    """
    x = np.asarray(x, dtype=np.float32)
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    y = (x - lo) / (hi - lo + 1e-6)
    y = np.clip(y, 0.0, 1.0)

    if abs(gamma - 1.0) > 1e-6:
        y = np.power(y, 1.0 / gamma)

    y8 = (y * 255.0).astype(np.uint8)

    if use_colormap and _HAVE_CV2:
        try:
            cmap = cv2.COLORMAP_MAGMA
        except AttributeError:
            cmap = cv2.COLORMAP_INFERNO
        y8 = cv2.applyColorMap(y8, cmap)
        y8 = cv2.cvtColor(y8, cv2.COLOR_BGR2RGB)
        return Image.fromarray(y8, mode="RGB")

    return Image.fromarray(y8, mode="L").convert("RGB")


def _error_map(pred_mask, gt_mask):
    # tp -> green, fp -> red, fn -> blue
    tp = (pred_mask & gt_mask)
    fp = (pred_mask & (~gt_mask))
    fn = ((~pred_mask) & gt_mask)
    rgb = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), np.uint8)
    rgb[..., 1] = (tp * 255).astype(np.uint8)
    rgb[..., 0] = (fp * 255).astype(np.uint8)
    rgb[..., 2] = (fn * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")

def _label(im: Image.Image, text: str):
    im = im.convert("RGB")
    im = ImageOps.expand(im, border=2, fill=(0, 0, 0))
    dr = ImageDraw.Draw(im)
    dr.text((6, 4), text, fill=(255, 255, 255))
    return im

def binarize(x, thr=0.5):
    return (x >= thr).astype(np.uint8)

def iou(a, b):
    inter = (a & b).sum()
    uni   = (a | b).sum()
    return float(inter) / (float(uni) + 1e-6)


# ----------------------------
# LiDAR -> BEV rasterizer (fixed metric window)
# ----------------------------
def lidar_hist_bev(points,
                   bev_hw=(128, 128),
                   meters=(60.0, 30.0),   # x forward in [0,xr], y lateral in [-yr/2, +yr/2]
                   z_range=(-3.0, 3.0),   # up/down band
                   fov_deg=180.0,         # front sector
                   blur_ksize=0,
                   dilate_ks=0,
                   log_scale=True):
    H, W = bev_hw
    bev = np.zeros((H, W), dtype=np.float32)
    if points is None or len(points) == 0:
        return torch.from_numpy(bev)[None, None].float()

    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return torch.from_numpy(bev)[None, None].float()

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]  # x fwd, y left, z up
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]

    if z_range is not None:
        zmin, zmax = z_range
        m = (z >= zmin) & (z <= zmax)
        x, y, z = x[m], y[m], z[m]

    if (fov_deg is not None) and (fov_deg < 360):
        ang = np.degrees(np.arctan2(y, x))
        half = fov_deg / 2.0
        m = (ang >= -half) & (ang <= half)
        x, y, z = x[m], y[m], z[m]

    if x.size == 0:
        return torch.from_numpy(bev)[None, None].float()

    xr, yr = meters
    x1, x2 = -1e-6, float(xr)
    y1, y2 = -float(yr) / 2, float(yr) / 2
    m = (x >= x1) & (x <= x2) & (y >= y1) & (y <= y2)
    x, y = x[m], y[m]
    if x.size == 0:
        return torch.from_numpy(bev)[None, None].float()

    gx = ((x - x1) / (x2 - x1) * (W - 1)).astype(np.int32)
    gy = ((y - y1) / (y2 - y1) * (H - 1)).astype(np.int32)
    bev[H - 1 - gy, gx] += 1.0  # flip y

    if log_scale:
        bev = np.log1p(bev)

    if blur_ksize and blur_ksize >= 3 and blur_ksize % 2 == 1 and _HAVE_CV2:
        bev = cv2.GaussianBlur(bev, (blur_ksize, blur_ksize), 0)

    mval = bev.max()
    if mval > 0:
        bev = bev / mval

    if dilate_ks and dilate_ks >= 3 and dilate_ks % 2 == 1 and _HAVE_CV2:
        kernel = np.ones((dilate_ks, dilate_ks), np.uint8)
        bev = cv2.dilate((bev * 255).astype(np.uint8), kernel, iterations=1).astype(np.float32) / 255.0

    return torch.from_numpy(bev)[None, None].float()


# ----------------------------
# Parameter-free lifter (voxel -> bilinear sample)
# ----------------------------
class ParameterFreeLifter(nn.Module):
    """
    Voxel-centric bilinear sampler:
      - Build a 3D voxel grid over ego-frame: x in [0,xr], z(lateral) in [-yr/2,+yr/2], y(up) in [ymin,ymax]
      - Project each voxel to image plane using K and T_cam_from_ego
      - Bilinearly sample features; reduce over vertical (mean)
    If calibration is missing, callers may fall back to pooling.
    """
    def __init__(self, bev_hw, meters, y_slices=1, y_min=0.0, y_max=0.0):
        super().__init__()
        self.bev_hw = bev_hw
        self.meters = meters
        self.y_slices = max(1, int(y_slices))
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self._has_calib = False
        self.register_buffer("_grid_uv", None, persistent=False)   # [Y,Hb,Wb,2]
        self.register_buffer("_valid_mask", None, persistent=False) # [Y,Hb,Wb,1]

    def load_calib(self, calib_json: str, image_hw: tuple):
        """
        calib_json format (single camera):
        {
          "cameras": [
            {
              "name": "cam0",
              "K": [[fx,0,cx],[0,fy,cy],[0,0,1]],
              "T_cam_from_ego": [[r11,...,tx],[...],[0,0,0,1]]
            }
          ]
        }
        """
        with open(calib_json, "r") as f:
            data = json.load(f)
        cams = data.get("cameras", [])
        if len(cams) == 0:
            self._has_calib = False
            return

        K = np.array(cams[0]["K"], dtype=np.float32)            # (3,3)
        T = np.array(cams[0]["T_cam_from_ego"], dtype=np.float32)  # (4,4)
        Himg, Wimg = image_hw
        self._build_sampling_grid(K, T, Himg, Wimg)
        self._has_calib = True

    def _build_sampling_grid(self, K, T_cam_from_ego, Himg, Wimg):
        Hb, Wb = self.bev_hw
        xr, yr = self.meters

        # voxel centers
        x = torch.linspace(0.0, xr, Wb)          # columns
        z = torch.linspace(-yr/2.0, yr/2.0, Hb)  # rows
        if self.y_slices == 1:
            y = torch.tensor([(self.y_min + self.y_max) * 0.5], dtype=torch.float32)
        else:
            y = torch.linspace(self.y_min, self.y_max, self.y_slices)

        # meshgrid order -> (Z rows, X cols, Y levels)
        Z, X, Y = torch.meshgrid(z, x, y, indexing="ij")  # [Hb, Wb, Y]
        ones = torch.ones_like(X)

        # Make [x, y, z, 1]^T in ego
        P_ego = torch.stack([X, Y, Z, ones], dim=-1)  # [Hb, Wb, Y, 4]
        P_ego = P_ego.reshape(-1, 4).t()              # [4, Hb*Wb*Y]

        # Ego->Camera
        T = torch.from_numpy(T_cam_from_ego)  # [4,4]
        P_cam = T @ P_ego                     # [4, N]
        Xc = P_cam[0, :]; Yc = P_cam[1, :]; Zc = P_cam[2, :]

        # project
        Kt = torch.from_numpy(K)              # [3,3]
        fx, fy = Kt[0,0], Kt[1,1]
        cx, cy = Kt[0,2], Kt[1,2]

        valid = Zc > 1e-3
        u = fx * (Xc / (Zc + 1e-6)) + cx
        v = fy * (Yc / (Zc + 1e-6)) + cy

        # normalize to [-1,1] for grid_sample
        u_n = (u / (Wimg - 1)) * 2 - 1
        v_n = (v / (Himg - 1)) * 2 - 1

        grid = torch.stack([u_n, v_n], dim=-1)  # [N,2]
        grid = grid.reshape(Hb, Wb, self.y_slices, 2).permute(2,0,1,3).contiguous()  # [Y,Hb,Wb,2]
        valid = valid.reshape(Hb, Wb, self.y_slices).permute(2,0,1).unsqueeze(-1).contiguous() # [Y,Hb,Wb,1]

        self._grid_uv = grid
        self._valid_mask = valid

    def forward(self, feat: torch.Tensor, image_hw: tuple):
        """
        feat: [B, C, Himg, Wimg]
        returns: [B, C, Hb, Wb] camera BEV
        """
        Hb, Wb = self.bev_hw
        if (self._grid_uv is None) or (self._valid_mask is None) or (not self._has_calib):
            # No calibration loaded; recommend falling back to pooling by caller
            return F.adaptive_avg_pool2d(feat, (Hb, Wb))

        device = feat.device
        grid = self._grid_uv.to(device)            # [Y,Hb,Wb,2]
        B, C, _, _ = feat.shape
        Ylev = grid.shape[0]

        outs = []
        for yi in range(Ylev):
            g = grid[yi:yi+1, ...].repeat(B, 1, 1, 1)  # [B,Hb,Wb,2]
            samp = F.grid_sample(feat, g, mode='bilinear', padding_mode='zeros', align_corners=True)  # [B,C,Hb,Wb]
            outs.append(samp)

        vol = torch.stack(outs, dim=2)  # [B, C, Y, Hb, Wb]
        bev = vol.mean(dim=2)           # reduce vertical
        return bev


# ----------------------------
# Dataset (+ light augmentation)
# ----------------------------
def _rand_rescale_crop(img: Image.Image, out_hw, scale_rng=(0.8, 1.2)):
    H, W = out_hw
    s = random.uniform(scale_rng[0], scale_rng[1])
    newH, newW = max(8, int(H * s)), max(8, int(W * s))
    img = img.resize((newW, newH), Image.BILINEAR)
    if newH >= H and newW >= W:
        top  = random.randint(0, newH - H)
        left = random.randint(0, newW - W)
        img = img.crop((left, top, left + W, top + H))
    else:
        canvas = Image.new("RGB", (W, H), (0, 0, 0))
        top  = random.randint(0, max(0, H - newH))
        left = random.randint(0, max(0, W - newW))
        canvas.paste(img, (left, top))
        img = canvas
    if random.random() < 0.5:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.5:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
    return img


class ToyPairsDataset(torch.utils.data.Dataset):
    """
    Loads image .jpg and LiDAR .npy; returns:
      img_t:   [3,H,W]
      aux_bev: [Caux,Hb,Wb]  (LiDAR BEV as aux input channels; currently Caux=1)
      tgt_bev: [1,Hb,Wb]     (LiDAR BEV supervision; clean/no blur/dilate)
    Expects: <root>/nuscenes_toy/images/*.jpg and .../lidar/*.npy
    """
    def __init__(self,
        root,
        limit=100000,
        img_size=(256, 448),
        bev_hw=(128, 128),
        augment=False,
        meters_x=60.0,
        meters_y=30.0,
        unit_scale=1.0,
        z_min=-3.0,
        z_max=3.0,
        fov_deg=180.0,
        blur_ksize=0,
        dilate_ks=0,
        swap_xy=False,
        flip_x=False,
        flip_y=False,
    ):
        super().__init__()
        self.root = Path(root)
        self.img_dir = self.root / "nuscenes_toy" / "images"
        self.lid_dir = self.root / "nuscenes_toy" / "lidar"
        self.imgs = sorted(list(self.img_dir.glob("*.jpg")) +
                           list(self.img_dir.glob("*.jpeg")) +
                           list(self.img_dir.glob("*.png")))
        self.pairs = []
        for p in self.imgs[:limit]:
            npy = self.lid_dir / f"{p.stem}.npy"
            if npy.exists():
                self.pairs.append((p, npy))

        self.img_size   = img_size
        self.bev_hw     = bev_hw
        self.augment    = augment

        self.meters_x   = float(meters_x)
        self.meters_y   = float(meters_y)
        self.unit_scale = float(unit_scale)
        self.z_min      = float(z_min)
        self.z_max      = float(z_max)
        self.fov_deg    = float(fov_deg)
        self.blur_ksize = int(blur_ksize)
        self.dilate_ks  = int(dilate_ks)

        self.swap_xy    = bool(swap_xy)
        self.flip_x     = bool(flip_x)
        self.flip_y     = bool(flip_y)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        imgp, npyp = self.pairs[idx]

        # image
        im = Image.open(imgp).convert("RGB")
        if self.augment:
            im = _rand_rescale_crop(im, self.img_size)
        else:
            im = im.resize(self.img_size[::-1], Image.BILINEAR)
        img_t = torch.from_numpy(np.array(im)).float().permute(2, 0, 1) / 255.0  # [3,H,W]

        # lidar points (npy Nx3 or Nx4)
        pts = np.load(npyp).astype(np.float32)
        if pts.ndim == 2 and pts.shape[1] >= 3:
            if self.unit_scale != 1.0:
                pts[:, :3] *= self.unit_scale
            if self.swap_xy:
                pts[:, [0, 1]] = pts[:, [1, 0]]
            if self.flip_x:
                pts[:, 0] = -pts[:, 0]
            if self.flip_y:
                pts[:, 1] = -pts[:, 1]
        else:
            pts = np.zeros((0, 3), dtype=np.float32)

        # --- AUX BEV (input to the network) ---
        aux_bev = lidar_hist_bev(
            pts,
            bev_hw=self.bev_hw,
            meters=(self.meters_x, self.meters_y),
            z_range=(self.z_min, self.z_max),
            fov_deg=self.fov_deg,
            blur_ksize=self.blur_ksize,   # optional smoothing for input
            dilate_ks=self.dilate_ks,
            log_scale=True,
        ).squeeze(0)  # [1,Hb,Wb]

        # --- TARGET BEV (clean supervision; NO blur/dilate) ---
        tgt_bev = lidar_hist_bev(
            pts,
            bev_hw=self.bev_hw,
            meters=(self.meters_x, self.meters_y),
            z_range=(self.z_min, self.z_max),
            fov_deg=self.fov_deg,
            blur_ksize=0,
            dilate_ks=0,
            log_scale=True,
        ).squeeze(0)  # [1,Hb,Wb]

        return img_t, aux_bev, tgt_bev


# ----------------------------
# Model (separate branches + fusion; no leakage)
# ----------------------------
class CamBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(96, 128, 3, stride=2, padding=1), nn.ReLU(),
        )
    def forward(self, x):  # [B,3,H,W] -> [B,128,H/16,W/16]
        return self.net(x)

class BEVResBlock(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(ch)
    def forward(self, x):
        r = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + r, inplace=True)

class SimpleBEV_XS(nn.Module):
    def __init__(self, bev_hw=(128,128), meters=(60.0,30.0),
                 lifter_mode="voxel", y_slices=1, y_min=0.0, y_max=0.0):
        super().__init__()
        self.bev_hw = bev_hw
        self.lifter_mode = lifter_mode
        self.cam = CamBackbone()

        # Voxel sampler (parameter-free) with optional calib
        self.lifter = ParameterFreeLifter(bev_hw=bev_hw, meters=meters,
                                          y_slices=y_slices, y_min=y_min, y_max=y_max)

        # Camera branch
        self.compress_cam = nn.Sequential(nn.Conv2d(128, 32, 3, padding=1), nn.ReLU())
        self.trunk_cam    = nn.Sequential(BEVResBlock(32), BEVResBlock(32))
        self.head_cam     = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 1, 1))

        # Aux (LiDAR/Radar) branch
        self.compress_aux = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU())
        self.trunk_aux    = nn.Sequential(BEVResBlock(32))
        self.head_aux     = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 1, 1))

        # Fusion
        self.trunk_fuse   = nn.Sequential(BEVResBlock(64), BEVResBlock(64))
        self.head_fuse    = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 1, 1))

        # dummy gate map for visualization compatibility
        self.register_buffer("_half_gate", None, persistent=False)

    def maybe_load_calib(self, calib_json: str, image_hw: tuple):
        if calib_json and Path(calib_json).exists():
            self.lifter.load_calib(calib_json, image_hw)

    def forward(self, img, aux_bev, image_hw=None, detach_aux: bool=False):
        """
        img:     [B,3,H,W]
        aux_bev: [B,1,Hb,Wb]
        returns: fused, cam_only, aux_only, gate_like
        """
        B, _, Himg, Wimg = img.shape
        feat = self.cam(img)  # [B,128,h,w]

        # lift to BEV (voxel sampler or pooling fallback)
        if self.lifter_mode == "pool":
            cam_bev = F.adaptive_avg_pool2d(feat, self.bev_hw)  # fallback
        else:
            cam_bev = self.lifter(feat, image_hw=(Himg, Wimg))   # voxel sampler

        # Separate branches (no leakage)
        cam32 = self.compress_cam(cam_bev)   # [B,32,Hb,Wb]
        aux32 = self.compress_aux(aux_bev)   # [B,32,Hb,Wb]

        cam_feat = self.trunk_cam(cam32)     # [B,32,Hb,Wb]
        aux_feat = self.trunk_aux(aux32)     # [B,32,Hb,Wb]
        if detach_aux:
            aux_feat = aux_feat.detach()     # block gradients from aux branch when requested

        fused   = torch.cat([cam_feat, aux_feat], dim=1)  # [B,64,Hb,Wb]
        fused   = self.trunk_fuse(fused)

        out_c = self.head_cam(cam_feat)   # camera-only head
        out_a = self.head_aux(aux_feat)   # aux-only head
        out_f = self.head_fuse(fused)     # fused head

        if self._half_gate is None or self._half_gate.shape != out_f.shape:
            self._half_gate = torch.full_like(out_f, 0.5)
        return out_f, out_c, out_a, self._half_gate


# ----------------------------
# Loss / Metrics
# ----------------------------
def bce_pos_weight(logits, target):
    with torch.no_grad():
        pos = target.sum()
        neg = target.numel() - pos
        pw = (neg / (pos + 1e-6)).clamp(1.0, 50.0)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)

def soft_dice_loss(logits, target, eps=1e-6):
    p = torch.sigmoid(logits)
    num = 2.0 * (p * target).sum(dim=(1,2,3))
    den = (p + target).sum(dim=(1,2,3)) + eps
    return 1.0 - (num / den).mean()


def eval_set(model, loader, device="cpu", thr=0.5, tb_thr=0.05, eval_cam_aux_off=False):
    """
    Evaluate fused, cam-only head, and aux-only head.
    If eval_cam_aux_off=True, the cam-only head is evaluated with AUX zeroed.
    """
    import numpy as np
    try:
        from sklearn.metrics import roc_auc_score
        have_auc = True
    except Exception:
        have_auc = False

    def acc():
        return dict(tp=0, fp=0, fn=0, iou_list=[], auc_sum=0.0, auc_n=0, lat_ms=[], N=0)

    Rf, Rc, Ra = acc(), acc(), acc()

    model.eval()
    for img, aux, tgt in loader:
        img, aux, tgt = img.to(device), aux.to(device), tgt.to(device)
        with torch.no_grad():
            # cam-only pass (optionally with aux OFF)
            t0 = time.time()
            if eval_cam_aux_off:
                _, out_c, _, _ = model(img, torch.zeros_like(aux), image_hw=(img.shape[2], img.shape[3]))
            else:
                _, out_c, _, _ = model(img, aux, image_hw=(img.shape[2], img.shape[3]))
            dt_cam = (time.time() - t0) * 1000.0

            # fused + aux-only with real aux
            t1 = time.time()
            out_f, _, out_a, _ = model(img, aux, image_hw=(img.shape[2], img.shape[3]))
            dt_fuse = (time.time() - t1) * 1000.0

        gt = (tgt.squeeze(1).cpu().numpy() >= tb_thr)

        def _upd(out, R, dt):
            prob = torch.sigmoid(out).squeeze(1).cpu().numpy()
            pred = (prob >= thr)

            R["tp"] += int((pred & gt).sum())
            R["fp"] += int((pred & (~gt)).sum())
            R["fn"] += int(((~pred) & gt).sum())

            for i in range(pred.shape[0]):
                inter = (pred[i] & gt[i]).sum()
                union = (pred[i] | gt[i]).sum()
                R["iou_list"].append(float(inter) / (float(union) + 1e-6))

            if have_auc:
                try:
                    R["auc_sum"] += roc_auc_score(gt.reshape(-1), prob.reshape(-1))
                    R["auc_n"] += 1
                except Exception:
                    pass

            R["lat_ms"].append(dt)
            R["N"] += pred.shape[0]

        _upd(out_c, Rc, dt_cam)
        _upd(out_a, Ra, dt_fuse)
        _upd(out_f, Rf, dt_fuse)

    def finish(R):
        tp, fp, fn = R["tp"], R["fp"], R["fn"]
        P = tp / (tp + fp + 1e-6)
        Rr = tp / (tp + fn + 1e-6)
        iou_micro = tp / (tp + fp + fn + 1e-6)
        iou_mean = float(np.mean(R["iou_list"])) if R["iou_list"] else 0.0
        auc = (R["auc_sum"] / R["auc_n"]) if R["auc_n"] > 0 else None
        lat = float(np.mean(R["lat_ms"])) if R["lat_ms"] else None
        return {
            "mIoU": iou_mean,
            "IoU_micro": iou_micro,
            "Precision": P,
            "Recall": Rr,
            "ROC_AUC": auc,
            "Latency_ms": lat,
            "N": R["N"],
            "TP": int(tp), "FP": int(fp), "FN": int(fn),
        }

    return finish(Rf), finish(Rc), finish(Ra)

def best_thr(model, loader, device="cpu", tb_thr=0.05, eval_cam_aux_off=False):
    """Grid-search a sigmoid threshold for the fused head using mIoU."""
    thrs = np.linspace(0.05, 0.95, 19)
    best = (0.0, 0.5)  # (best_mIoU, thr)
    for t in thrs:
        fusion, _, _ = eval_set(model, loader, device=device, thr=t, tb_thr=tb_thr, eval_cam_aux_off=eval_cam_aux_off)
        if fusion["mIoU"] > best[0]:
            best = (fusion["mIoU"], t)
    return best


# ----------------------------
# Visualization
# ----------------------------
@torch.no_grad()
def save_val_viz(model, loader, out_dir, thr=0.5, tb_thr=0.2, device="cpu",
                 nmax=8, p_low=1.0, p_high=99.0, gamma=0.7,
                 viz_dilate=0, use_cmap=False):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0

    def to_vis(arr):
        a = np.asarray(arr, dtype=np.float32)
        if viz_dilate and viz_dilate >= 3 and (viz_dilate % 2 == 1) and _HAVE_CV2:
            k = np.ones((viz_dilate, viz_dilate), np.uint8)
            a = cv2.dilate(a, k, iterations=1)
        return _heat_to_img(a, p_low=p_low, p_high=p_high, gamma=gamma, use_colormap=use_cmap)

    for img, aux, tgt in loader:
        img = img.to(device); aux = aux.to(device); tgt = tgt.to(device)
        out_f, out_c, out_a, gate = model(img, aux, image_hw=(img.shape[2], img.shape[3]))
        prob_f = torch.sigmoid(out_f).squeeze(1).cpu().numpy()
        prob_c = torch.sigmoid(out_c).squeeze(1).cpu().numpy()
        prob_a = torch.sigmoid(out_a).squeeze(1).cpu().numpy()
        tgt_np = tgt.squeeze(1).cpu().numpy()
        gate_w = gate.squeeze(1).cpu().numpy()

        B = img.shape[0]
        for b in range(B):
            if saved >= nmax:
                return
            cam = (img[b].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            cam_img  = Image.fromarray(cam, mode="RGB")

            aux_img  = to_vis(aux[b,0].detach().cpu().numpy())
            gt_img   = to_vis(tgt_np[b])
            pf_img   = to_vis(prob_f[b])
            pc_img   = to_vis(prob_c[b])
            pa_img   = to_vis(prob_a[b])
            gate_img = to_vis(gate_w[b])

            pm = (prob_f[b] >= thr); gm = (tgt_np[b] >= tb_thr)
            err_img = _error_map(pm, gm)
            inter = np.logical_and(pm, gm).sum()
            uni   = np.logical_or(pm, gm).sum()
            iou_v = float(inter) / (float(uni) + 1e-6)

            tiles = [
                _label(cam_img, "Camera"),
                _label(aux_img, "Aux BEV (LiDAR/Radar)"),
                _label(gt_img,  "BEV GT (LiDAR)"),
                _label(pf_img,  f"Fused BEV (thr={thr:.2f})"),
                _label(pc_img,  "Cam-only head"),
                _label(pa_img,  "Aux-only head"),
                _label(gate_img,"Gate-like (0.5)"),
                _label(err_img, f"Error (IoU={iou_v:.3f})"),
            ]
            h = max(t.height for t in tiles)
            tiles = [t.resize((t.width, h)) for t in tiles]
            W = sum(t.width for t in tiles)
            strip = Image.new("RGB", (W, h), (0,0,0))
            x = 0
            for t in tiles:
                strip.paste(t, (x, 0)); x += t.width
            strip.save(out_dir / f"val_{saved:02d}_panel.png")
            saved += 1


# ----------------------------
# Main
# ----------------------------
def main():
    set_seed(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Datasets root; expects nuscenes_toy/{images,lidar}")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])

    # image size
    ap.add_argument("--img_h", type=int, default=256)
    ap.add_argument("--img_w", type=int, default=448)

    # BEV / rasterizer
    ap.add_argument("--bev_h", type=int, default=128)
    ap.add_argument("--bev_w", type=int, default=128)
    ap.add_argument("--meters_x", type=float, default=60.0)
    ap.add_argument("--meters_y", type=float, default=30.0)
    ap.add_argument("--unit_scale", type=float, default=1.0)  # safer default for your dataset
    ap.add_argument("--z_min", type=float, default=-3.0)
    ap.add_argument("--z_max", type=float, default=3.0)
    ap.add_argument("--fov_deg", type=float, default=180.0)
    ap.add_argument("--swap_xy", action="store_true")
    ap.add_argument("--flip_x", action="store_true")
    ap.add_argument("--flip_y", action="store_true")
    ap.add_argument("--blur_ksize", type=int, default=0, help="Gaussian blur kernel (odd ≥3). 0=off (aux only)")
    ap.add_argument("--dilate_ks", type=int, default=0, help="Dilation kernel (odd ≥3). 0=off (aux only)")
    ap.add_argument("--stats", action="store_true", help="Print BEV statistics (shape/min/max/mean/nonzero).")

    # lifter & calib
    ap.add_argument("--lifter", choices=["voxel","pool"], default="voxel",
                    help="voxel: parameter-free bilinear sampling; pool: adaptive pooling")
    ap.add_argument("--calib", type=str, default="", help="path to calib.json (K & T_cam_from_ego)")
    ap.add_argument("--y_slices", type=int, default=1, help="vertical slices (>=1). 1 ~ ground-plane")
    ap.add_argument("--y_min", type=float, default=0.0)
    ap.add_argument("--y_max", type=float, default=0.0)

    # evaluation controls
    ap.add_argument("--tb_thr", type=float, default=0.2, help="GT binarization threshold for evaluation/viz.")
    ap.add_argument("--eval_cam_aux_off", action="store_true", help="Zero aux input when evaluating camera-only.")

    # visualization controls
    ap.add_argument("--viz_p_low",  type=float, default=1.0,  help="Low percentile for contrast stretch.")
    ap.add_argument("--viz_p_high", type=float, default=99.0, help="High percentile for contrast stretch.")
    ap.add_argument("--viz_gamma",  type=float, default=0.7,  help="Gamma (<1 brightens).")
    ap.add_argument("--viz_dilate", type=int,   default=0,    help="Dilate kernel (odd>=3) for viz only.")
    ap.add_argument("--viz_colormap", action="store_true",    help="Use a perceptual colormap for BEV tiles.")

    # training
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--optim", default="adamw", choices=["adam","adamw"])
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--onecycle", action="store_true")

    # --- Fusion de-biasing / robustness ---
    ap.add_argument("--camonly_w",     type=float, default=0.5,
                    help="Weight for fused loss from a cam-only pass (AUX=0).")
    ap.add_argument("--cam_head_w",    type=float, default=0.10,
                    help="Weight for camera-only head loss.")
    ap.add_argument("--aux_head_w",    type=float, default=0.02,
                    help="Weight for aux-only head loss (keep small).")
    ap.add_argument("--aux_dropout_p", type=float, default=0.30,
                    help="Per-batch prob to zero the aux input (sensor dropout).")
    ap.add_argument("--aux_noise_std", type=float, default=0.05,
                    help="Stddev of Gaussian noise added to aux input.")
    ap.add_argument("--detach_aux_p",  type=float, default=0.30,
                    help="Per-batch prob to detach aux features in fused pass.")

    args = ap.parse_args()

    # robust: ensure output dir exists now and reuse Path
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if (args.device in ["auto","cuda"] and torch.cuda.is_available()) else "cpu"
    print("Device:", device)

    bev_hw = (args.bev_h, args.bev_w)
    common_ds_kwargs = dict(
        img_size=(args.img_h, args.img_w),
        bev_hw=bev_hw,
        meters_x=args.meters_x,
        meters_y=args.meters_y,
        unit_scale=args.unit_scale,
        z_min=args.z_min, z_max=args.z_max,
        fov_deg=args.fov_deg,
        blur_ksize=args.blur_ksize,
        dilate_ks=args.dilate_ks,
        swap_xy=args.swap_xy,
        flip_x=args.flip_x,
        flip_y=args.flip_y,
    )

    # build once to know pairs; then split
    ds_all = ToyPairsDataset(args.root, limit=100000, augment=False, **common_ds_kwargs)
    if len(ds_all) == 0:
        print("[!] No pairs found. Expected data under", Path(args.root)/"nuscenes_toy")
        return

    N = len(ds_all)
    split = max(1, int(0.8 * N))
    ds_train = ToyPairsDataset(args.root, limit=100000, augment=True,  **common_ds_kwargs)
    ds_val   = ToyPairsDataset(args.root, limit=100000, augment=False, **common_ds_kwargs)
    ds_train.pairs = ds_all.pairs[:split]
    ds_val.pairs   = ds_all.pairs[split:]
    print(f"Train {len(ds_train)} | Val {len(ds_val)}")
    if len(ds_train) == 0 or len(ds_val) == 0:
        print("Not enough samples."); return

    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=args.batch, shuffle=True,  drop_last=False, num_workers=2)
    val_loader   = torch.utils.data.DataLoader(ds_val,   batch_size=args.batch, shuffle=False, drop_last=False, num_workers=2)

    model = SimpleBEV_XS(
        bev_hw=bev_hw,
        meters=(args.meters_x, args.meters_y),
        lifter_mode=args.lifter,
        y_slices=args.y_slices,
        y_min=args.y_min,
        y_max=args.y_max
    ).to(device)

    # load calib if provided (assumes fixed calib across dataset)
    if args.calib:
        model.maybe_load_calib(args.calib, image_hw=(args.img_h, args.img_w))

    if args.optim == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.onecycle:
        steps_per_epoch = max(1, len(train_loader))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch)
    else:
        sched = None

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device=="cuda"))

    # Train
    for ep in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"ep{ep}")
        opt.zero_grad(set_to_none=True)
        for it, (img, aux, tgt) in pbar:
            img = img.to(device)
            aux = aux.to(device)
            tgt = tgt.to(device)

            # ---- Build a degraded AUX input (dropout + noise) ----
            aux_in = aux.clone()
            if random.random() < args.aux_dropout_p:
                aux_in.zero_()  # simulate missing sensor
            if args.aux_noise_std > 0:
                aux_in = (aux_in + torch.randn_like(aux_in) * args.aux_noise_std).clamp_(0, 1)
            use_detach = (random.random() < args.detach_aux_p)

            with torch.cuda.amp.autocast(enabled=(args.amp and device=="cuda")):
                # PASS A: fused with (possibly degraded) AUX
                out_f, out_c, out_a, _ = model(
                    img, aux_in, image_hw=(img.shape[2], img.shape[3]), detach_aux=use_detach
                )
                # PASS B: fused with AUX forcibly OFF (camera-only fused)
                out_f_camonly, _, _, _ = model(
                    img, torch.zeros_like(aux), image_hw=(img.shape[2], img.shape[3]), detach_aux=False
                )

                # Losses
                def _seg_loss(logits):
                    return bce_pos_weight(logits, tgt) + 0.5 * soft_dice_loss(logits, tgt)

                loss_f         = _seg_loss(out_f)
                loss_f_camonly = _seg_loss(out_f_camonly)
                loss_cam_head  = (bce_pos_weight(out_c, tgt) + soft_dice_loss(out_c, tgt)) * args.cam_head_w
                loss_aux_head  = (bce_pos_weight(out_a, tgt) + soft_dice_loss(out_a, tgt)) * args.aux_head_w

                # Total objective: good with aux AND good without aux
                loss = loss_f + args.camonly_w * loss_f_camonly + loss_cam_head + loss_aux_head

            scaler.scale(loss).backward()
            if ((it + 1) % args.accum == 0) or (it + 1 == len(train_loader)):
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                if sched is not None: sched.step()
            pbar.set_postfix(loss=float(loss.item()))

        # Eval + sigmoid threshold search on fused head
        best_iou, thr = best_thr(model, val_loader, device=device, tb_thr=args.tb_thr, eval_cam_aux_off=args.eval_cam_aux_off)
        fusion, camera, aux_only = eval_set(model, val_loader, device=device, thr=thr, tb_thr=args.tb_thr, eval_cam_aux_off=args.eval_cam_aux_off)
        print(f"\n[ep{ep}] best_thr={thr:.2f} | Fusion {fusion} | Cam {camera} | Aux {aux_only}")

    # Save metrics (ensure dir exists just before writing)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"best_thr": float(thr), "fusion": fusion, "camera": camera, "aux": aux_only}, f, indent=2)
    print("Saved", out_dir / "metrics.json")

    # Qualitative panels
    viz_dir = out_dir / "viz"
    save_val_viz(
        model, val_loader, viz_dir, thr=thr, tb_thr=args.tb_thr, device=device, nmax=8,
        p_low=args.viz_p_low, p_high=args.viz_p_high, gamma=args.viz_gamma,
        viz_dilate=args.viz_dilate, use_cmap=args.viz_colormap
    )
    print("Wrote qualitative panels to", viz_dir)


if __name__ == "__main__":
    main()
