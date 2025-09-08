import json, pprint
with open("outputs/maritime_8pairs/metrics.json") as f:
    m = json.load(f)
pprint.pp(m)

# example: grab best IoU threshold and fused mIoU
print("best_thr =", m["best_thr"])
print("fused mIoU =", m["fusion"]["mIoU"])

