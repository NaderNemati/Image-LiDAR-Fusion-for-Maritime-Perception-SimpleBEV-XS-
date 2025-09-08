<div align="center">
  <table border=0 style="border: 0px solid #c6c6c6 !important; border-spacing: 0px; width: auto !important;">
    <tr>
      <td valign=top style="border: 0px solid #c6c6c6 !important; padding: 0px !important;">
        <div align=center valign=top>
          <img src="https://github.com/NaderNemati/Image-LiDAR-Fusion-for-Maritime-Perception-SimpleBEV-XS-/blob/main/c53465dc-5122-40d3-a7f4-1aab67cac584.png" style="margin: 0px !important; height: 400px !important;">
        </div>
      </td>
    </tr>
  </table>
</div>

---

A practical, calibration-lean BEV fusion pipeline for maritime scenes.
It time-aligns monocular RGB with forward LiDAR, rasterizes points into BEV, and trains a compact fusion head for occupancy/obstacle likelihood over water.

Latest validation(e.g. N = 4): Fusion mIoU 0.93 / 0.90 / 0.94 / 0.94 per panel at thr=0.60 (qualitative set), and mIoU 0.837 @ best_thr 0.95 from metrics.json. Best sweep epoch: mIoU 0.866 @ thr 0.75 with Recall ≈ 0.992 and strong ROC-AUC.


- [Installation](#installation)
- [Usage](#usage)
- [Project Overview](#Project_Overview)
- [Resource Management for CPU and GPU Allocation](Resource_Management_for_CPU_and_GPU_Allocation)
- [Modular and Configuration](#Modular_and_Configuration)
- [License](#license)
- [Summary of Analysis of Model Performance](#Summary_of_Analysis_of_Model_Performance)





# Installation

### Prerequisites

Python 3.9+

PyTorch (CUDA optional)

OpenCV, NumPy, SciPy, Matplotlib

(Optional) cuDNN/CUDA for GPU acceleration

```bash
git clone https://github.com/<your-org-or-user>/<repo-name>.git
cd <repo-name>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
