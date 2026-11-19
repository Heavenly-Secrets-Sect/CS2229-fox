# CS2229-fox (`foxrepro`)

Tái lập và kiểm toán cơ chế của **Fox** (ICML 2026, "Dismantling Pathological Shortcuts...") trên LLaVA-1.5, hai benchmark POPE và CHAIR. Máy đích: Kaggle, 2 GPU Tesla T4. **Đọc `docs/CATCHUP.md` trước khi làm gì**: nó giải thích thuật ngữ (Fox, POPE, CHAIR, ULP, TOST...), tóm tắt đã chạy gì và biết được gì, và liệt kê những lần đã sai để không lặp lại. Đồ án môn CS2229.

## Lệnh
- Test (stdlib unittest, pytest chỉ là runner tùy chọn): `PYTHONPATH=src python3 -m unittest discover -s tests -t .` (khoảng 400 test; phải xanh trước mọi commit).
- Lập kế hoạch round: `python3 -m foxrepro.bench.plan plan --budget-hours 10 --round <rN>`; nhận kết quả từ `runs_inbox/`: `python3 -m foxrepro.bench.plan ingest`.
- Một session trên Kaggle: `bash run_kaggle_fox.sh <N>` (N từ 1 đến số session); `DRY=1` chỉ in lệnh.
- Chạy trên Kaggle qua CI: publish một release; `kaggle-run.yml` khai báo (notebook `ml-fox-ci`). Round do bot `message-talisman` điều phối. Tải zip kết quả về `runs_inbox/`: note KB `70_Van-hanh/kaggle-tai-ket-qua.md`.

## Quy tắc cứng
- **`sessions` trong `kaggle-run.yml` phải khớp `scripts/run_plan.json`.** Một N ngoài phạm vi là một session chạy rỗng, tốn quota GPU. `tests/test_plan.py::TestTheKaggleCiManifest` giữ hai file đồng bộ; đóng băng plan lại thì sửa cả hai.
- **Không gọi `kaggle_setup.sh` trong CI.** Nó dựng Track A (transformers 4.29, Python 3.11) cho các arm đối chiếu với release gốc; image Kaggle hiện tại là Python 3.13 nên không dựng được. Track B (transformers của host) là cái đã sinh ra mọi record trong `results/runs`.
- **`results/runs/` được track có chủ đích** (khoảng 9 MB cho 53 job; xem `results/runs/README.md`). Đừng sửa tay hay xóa các record đã commit. Các thư mục chạy trực tiếp (`/runs/`, logs) là trạng thái tạm và bị ignore.
- Thay đổi so với kế hoạch hay tiêu chí ghi vào `docs/DEVIATIONS.md`, không im lặng đổi sau khi đã có kết quả.
- Có test kiểm tra tài liệu không lỗi thời (`tests/test_docs_are_not_stale.py`): sửa tài liệu thì chạy lại test.
- Không commit `.env`, token, zip kết quả từ Kaggle (`runs_inbox/` bị ignore), hay PDF paper của người khác.
- Không push, publish release (tốn quota GPU của thành viên) hay xóa kết quả khi chưa được hỏi.

## Điều cần biết
- Các round được chạy bởi bot trên nhiều account Kaggle; một session có thể lỗi một phần mà vẫn có zip (đọc `_job_times.tsv` trong zip). Session bị cắt vẫn để lại zip của mọi job đã xong vì `pack_runs.sh` đóng gói sau mỗi job.
- `FOX_PERSIST_ROOT` đẩy runs và logs sang `/kaggle/temp` để `/kaggle/working` chỉ còn một zip (thứ nút tải trỏ tới).

## Liên kết
- Note dự án trong vault KB: `~/Desktop/Workspace/knowledge-base/20_Nghien-cuu/cs2229-fox/cs2229-fox-index.md`
- Sổ đăng ký project (cục bộ): `~/Desktop/Workspace/PROJECTS.md`
