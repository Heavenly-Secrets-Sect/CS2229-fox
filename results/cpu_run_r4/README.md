# Caption sinh trên CPU — r4 session 1 và 2 (2026-10-25)

106 caption CHAIR greedy sinh ra trên **CPU**, kèm bản **GPU** của đúng cùng
sample để so. Đây không phải kết quả dùng cho bảng nào; nó là bằng chứng đo
được, và là bản sao duy nhất sau khi hai zip `fox_runs_fox_r4_s{1,2}.zip` bị
xoá.

## Chúng từ đâu ra

Hai session Kaggle r4 chạy mười tiếng trên notebook **chưa gắn accelerator**.
`torch 2.11.0+cpu`, `n_gpus=0`. Mỗi caption tốn **654 s** và **632 s** thay vì
7.8 s trên 2×T4 — chậm 84 lần — nên hai session mười tiếng chỉ ra được 52 và
54 caption trong số 500. Hai mươi giờ quota không mua được gì.

Sau đó `_refuse_cpu_only_host` ra đời (`fox/attention.py`) để chặn, cùng với
`scripts/require_gpu.py` trong runner. Arm này đã được chạy lại trọn vẹn trên
GPU; `results/runs/` giữ bản GPU, 500/500 caption, toàn bộ ở commit `7d52f44`,
không trộn phiên bản.

## Vì sao đáng giữ

Greedy là tiền định. Cùng code, cùng seed, cùng sample — **khác phần cứng**:

| | |
|---|---|
| caption CPU | 106 |
| lệch so với bản GPU | **11 (10.4 %)** |

Code giữa hai lần chạy khác nhau đúng một guard (`434b004` → `7d52f44`,
53 dòng thêm, không chạm tensor nào), nên 10.4 % này là **sàn tái lập của
CHAIR khi đổi phần cứng**, không phải hiệu ứng của một thay đổi thuật toán.

Con số đó đáng mang theo khi đọc mọi khoảng cách CHAIR trong dự án: gap của
Fox là 6.98 điểm CHAIR_S, và riêng việc đổi CPU/GPU đã lật một phần mười số
caption. Nó cùng họ với phát hiện trung tâm của dự án — hiệu ứng của Fox là
phép cast logit chứ không phải can thiệp attention: cả hai đều nói rằng đầu ra
của LVLM nhạy với nhiễu số học ở mức thấp hơn hẳn mức mà phương pháp tuyên bố
tác động.

## Định dạng

JSONL, mỗi dòng là một record chạy CPU, thêm hai khoá:

* `_gpu_output` — output của cùng `sample_id` trong bản GPU đã lưu
* `_differs_from_gpu` — `true` ở đúng 11 dòng
