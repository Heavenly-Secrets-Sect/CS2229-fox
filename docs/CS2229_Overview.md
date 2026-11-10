# Tổng quan đồ án — tái lập và kiểm toán Fox (ICML 2026)

Tài liệu này dành cho người **mới tham gia đồ án**. Đọc hết là đủ hiểu đồ án
làm gì, đang ở đâu, và bạn nhận được việc gì. Muốn đi sâu thì có bản đồ ở
mục 9.

---

## 1. Đồ án làm gì, trong một đoạn

Paper **Fox** (Liu Yu et al., ICML 2026, PMLR v306 — *"Dismantling Pathological
Shortcuts: A Causal Framework for Faithful LVLM Decoding"*) nói rằng mô hình
thị giác–ngôn ngữ bịa ra vật thể không có trong ảnh vì một số **attention head
cụ thể** ở các bước giải mã quan trọng tách khỏi bằng chứng thị giác và bám vào
tiên nghiệm ngôn ngữ neo ở system prompt. Fox đề xuất chẩn đoán các head đó,
can thiệp vào attention logit của chúng, rồi hợp nhất hai nhánh.

Đồ án này **tái lập** các con số đó trên phần cứng của mình, và **kiểm toán**
xem cơ chế mà paper mô tả có thật là thứ tạo ra hiệu ứng hay không.

Hai việc đó khác nhau, và sự khác nhau chính là đóng góp của đồ án: chúng ta
tái lập được hiệu ứng nhưng **không** tái lập được lời giải thích.

## 2. Vì sao đồ án này đáng làm

Một bài tái lập chỉ xác nhận "số có ra giống không" thì giá trị thấp. Chỗ có
giá trị là: một phương pháp có thể cho ra đúng con số đã công bố **vì một lý do
hoàn toàn khác** với lý do mà paper viết. Khi đó bảng số liệu không phát hiện
được gì — phải thiết kế thí nghiệm riêng để phát hiện.

Đó là điều đã xảy ra ở đây, và nó xảy ra ở **ba** phương pháp khác nhau (Fox,
OPERA, CausalMM). Xem mục 5.

## 3. Phạm vi — và những gì nằm ngoài

**Phần cứng:** Kaggle Notebook, 2× Tesla T4 16 GB, quota ~30 GPU-h/tuần. Mỗi
session tối đa 12h nên ta luôn lên kế hoạch dưới 11h.

| | |
|---|---|
| **Backbone** | `llava-hf/llava-1.5-7b-hf`, fp16, `device_map="auto"` (shard qua 2 GPU) |
| **Benchmark** | POPE (3 split × 3000 câu hỏi yes/no) và CHAIR (500 ảnh COCO val2014, sinh caption) |
| **Phương pháp** | `sampling` (baseline), `fox`, `vcd`, `icd`, `sid`, `causalmm`, `opera` |

**Nằm ngoài, và vì sao** — ghi rõ để báo cáo nói được thay vì để lỗ trống:

- **InstructBLIP / Shikra** (cột 2 và 3 của mọi bảng chính). InstructBLIP chạy
  một *bản copy thứ hai* của phần can thiệp trong release, bỏ `m_sys` và dùng
  layer 4–10 với k=0.4 — là một phương pháp khác, không phải đổi tham số.
  Shikra không có bản chuyển đổi HF được bảo trì.
- **MME / MMBench / MathVista** (Figure 5 và phụ lục). Cần harness viết mới;
  đáng chú ý là release của Fox **không hề có script MME** dù báo cáo kết quả
  MME.
- **GPT-4V judge** (Figure 6). Cần API key trả phí. Protocol nằm ở Appendix
  C.3. Đang tạm hoãn theo quyết định của nhóm.

Không bao giờ dùng **checkpoint lượng tử hoá** (gptq-4bit, 8bit). Đại lượng
đang nghiên cứu *chính là* hiệu ứng độ chính xác số học, nên lượng tử hoá sẽ
phá đúng thứ ta đang đo.

## 4. Nguyên tắc thiết kế quan trọng nhất

Năm trong sáu phương pháp so sánh là **tự implement** từ source của từng
paper. Nên repo có một cấu trúc hai tầng mà bạn cần hiểu trước khi sửa bất kỳ
dòng nào:

```
src/foxrepro/fox/reference.py   ← chép NGUYÊN VĂN từ code đã phát hành
src/foxrepro/fox/kernel.py      ← bản port sạch, test khẳng định TRÙNG BIT
```

`reference.py` là **oracle số học**. Nó xấu, nó giữ nguyên cả những chỗ code
gốc không khớp với văn bản paper. Đó là chủ ý: nó trả lời câu hỏi *"code của ta
có bằng code của họ?"*.

Nhưng nó **không** trả lời được hai câu hỏi khác:

1. *Code của họ có bằng paper của họ?* — Thường là không. Ví dụ: harness của
   SID không chạy nổi VCD; sampler hai-counterfactual của CausalMM là dead
   code; nửa retrospection của OPERA không thể kích hoạt trên POPE.
2. *Cơ chế có thực sự hoạt động trên model thật?* — Một phương pháp contrastive
   mà nhánh âm của nó **không** tệ hơn vẫn cho ra một con số accuracy. Một
   phương pháp mà quy tắc của nó chưa bao giờ đổi một quyết định nào cũng vậy.
   Không bảng nào nhìn thấy điều đó.

Nên có tầng thứ ba: [`docs/VALIDATION.md`](VALIDATION.md) — **6 dự đoán được
đăng ký trước** (V1–V6), viết xuống *trước khi* chạy, mỗi dự đoán kèm cách đọc
kết quả cho mọi khả năng. Hiện trạng: **18 dự đoán đúng, 1 sai.**

> Đây là nguyên tắc văn hoá của đồ án: nếu một thí nghiệm có thể được giải
> thích theo nhiều cách sau khi thấy số, thì phải chọn cách giải thích
> **trước**. Cái sai duy nhất tính đến nay là một dự đoán của chúng ta, không
> phải một bug — và nó được ghi lại đúng như vậy.

## 5. Phát hiện chính

### 5.1 Về Fox — hiệu ứng tái lập được, lời giải thích thì không

POPE-adversarial, sampling, 10 seed:

| arm | gap vs baseline | TOST vs `fox` (Δ=±0.5) |
|---|---|---|
| `fox` | +5.03 [+4.68, +5.38] | — |
| `fox ulp_control` | +5.11 [+4.91, +5.30] | **tương đương**, p=0.0154 |
| `fox logit_dtype=bf16` | +5.05 [+4.69, +5.41] | **tương đương**, p=0.0035 |
| `fox off` | +2.21 [+1.84, +2.58] | không, p=1.000 |
| `fox logit_dtype=fp32` (greedy, n=3000) | **+0.03 [−0.06, +0.09]** | hiệu ứng **biến mất** |

Đọc bảng này từ dưới lên:

- `ulp_control` **chỉ làm tròn** logit, không trừ γ, không can thiệp attention
  — và tương đương thống kê với Fox đầy đủ.
- `logit_dtype=fp32` **giữ nguyên** can thiệp attention nhưng bỏ phép làm tròn
  — và hiệu ứng về 0.03 điểm.

Vậy cơ chế là **phép cast logit**, không phải can thiệp attention.

Thêm hai điểm:

- **Cái paper *mô tả* không chạy.** Arm `as_described` — làm đúng Appendix B —
  chỉ được +0.40 điểm. Cái code *làm* thì chạy.
- **Tiền đề nhân quả không đứng.** Bỏ system prompt đi thì gain **tăng**
  (+4.37 vs +3.33), trong khi `m_sys` được cho là gốc của đường tắt. Và
  `boost_visual` — tăng attention ảnh thật, đúng thứ câu chuyện của paper
  khuyến nghị và paper bỏ qua — **thắng Fox trên CHAIR** (−13.22 vs −6.98
  CHAIR_S) mà **không mất recall**.

### 5.2 Về OPERA — con số POPE của nó là beam search

OPERA là ba cơ chế xếp lên nhau nhưng paper chỉ đặt tên cho một: (1) beam
search 5 beam, (2) hạn chế cứng 5 candidate, (3) hình phạt "over-trust" thưởng
cho việc chú ý vào ảnh. Ta phân rã cả ba:

| arm | accuracy | vs OPERA |
|---|---|---|
| `opera` | 79.10 | — |
| `opera_no_penalty` | 79.03 | −0.07 |
| `opera_unrestricted` | 79.13 | +0.03 |

0.10 điểm trên 3000 câu. Telemetry nói rõ vì sao: độ tản của hình phạt trên 5
candidate là **0.067** so với độ tản logit **9.099** — nhỏ hơn 136 lần — và
trên 3000 câu nó đổi top candidate ở **đúng 1 bước**.

### 5.3 Về CausalMM — counterfactual thị giác là một mô tả của thứ code không làm

Mỗi hàng attention của CLIP là softmax trên 577 key, nên ma trận tổng bằng 577
và trung bình mỗi ô là `1/577`. Shuffle một ma trận **không đổi được trung
bình**, và bilinear resize hay mean-over-heads cũng không. Hệ quả: "nhánh
counterfactual" chỉ là ảnh bị hạ tỉ lệ ~580 lần, tức gần như mù.

Đo trên model thật, dưới greedy, 500 câu:

| arm | trùng `shuffle` | accuracy |
|---|---|---|
| `cf_constant` (thay bằng đúng `1/n_keys`) | 498/500 | 77.80 |
| `cf_blind` (xoá trắng feature) | 495/500 | **78.80** |

Hàng đầu xác nhận lập luận. Hàng thứ hai là dự đoán **sai** duy nhất của đồ án
— và sai theo hướng thú vị: `cf_blind` đúng trên **5/5** câu nó khác `shuffle`,
toàn bộ là label `no` mà hai arm kia bịa `Yes`. Nhánh mù hoàn toàn là tiên
nghiệm ngôn ngữ thuần nhất, nên trừ nó đi hiệu quả hơn. **CausalMM sẽ điểm cao
hơn nếu nó làm ít hơn.** (McNemar chính xác p=0.0625 — đang nâng lên 3000 câu
để chốt.)

## 6. Tình trạng hiện tại

- **238/238 job trong plan đã chạy** (3 job đang mở rộng lên full split ở
  session `fox_r11_s1`).
- Gate kiểm chứng: **18 đúng / 1 sai**.
- Mọi **baseline** POPE khớp paper trong 0.05 điểm → harness, dữ liệu, protocol
  đều đúng. Những gì đo được là về phương pháp, không phải về setup.

Tái lập so với paper:

| artefact | của mình [CI95] | paper | kết luận |
|---|---|---|---|
| POPE-adv, sampling (10 seed) | +5.03 [+4.68, +5.38] | +5.16 | **tái lập** |
| POPE-random, sampling (3 seed) | +4.16 [+3.86, +4.45] | +4.20 | **tái lập** |
| POPE-popular, sampling (3 seed) | +4.13 [+2.63, +5.64] | +4.17 | điểm khớp, chưa phân giải sau Holm |
| POPE ×3 split, greedy | +0.20 / +1.00 / +3.33 | +0.10 / +1.14 / +3.33 | **tái lập** |
| CHAIR_S, sampling (10 seed) | −6.98 [−9.25, −4.71] | −11.20 | **không** |
| CHAIR_I, sampling (10 seed) | −2.67 [−3.37, −1.96] | −4.28 | **không** |
| Table 6 (tỉ lệ latency) | 2.03× | 1.87× | **tái lập** |

CHAIR không tái lập, và lý do đã đo được: khoảng cách dịch **5.20 điểm** qua
năm lần rút tập ảnh khác nhau, nên khoảng tin cậy một-lần-rút đánh giá thấp độ
bất định. Tập ảnh gốc của paper không thể phục hồi.

### So sánh bảy phương pháp (POPE-adversarial, protocol dùng chung)

Sampling, trung bình qua seed:

| phương pháp | n seed | acc | F1 | gap |
|---|---|---|---|---|
| `fox` | 10 | 81.76 | 82.13 | **+5.03** |
| `sid` | 3 | 79.67 | 80.75 | +2.94 |
| `vcd_fox_fusion` | 3 | 78.81 | 79.07 | +2.09 |
| `causalmm` | 3 | 77.86 | 80.13 | +1.13 |
| `icd` | 3 | 77.20 | 79.15 | +0.48 |
| `sampling` (baseline) | 10 | 76.72 | 78.93 | — |
| `vcd` | 3 | 75.80 | 78.34 | **−0.92** |

Greedy, n=3000, một lần chạy mỗi arm (tiền định nên một lần là đủ):

| phương pháp | acc | F1 | gap |
|---|---|---|---|
| `fox` | 82.37 | 82.65 | +3.33 |
| `opera`, protocol 10 token của OPERA | 81.13 | 81.88 | **+2.10** |
| `sid` | 80.23 | 81.29 | +1.20 |
| `opera`, harness 1 token của Fox | 79.10 | 80.95 | +0.07 |
| `sampling` (baseline) | 79.03 | 80.89 | — |
| `icd` | 78.60 | 80.41 | −0.43 |
| `vcd` | 76.93 | 79.08 | −2.10 |

Hai chỗ bắt buộc phải chú ý khi đọc hai bảng này:

- **`vcd` và `vcd_fox_fusion` là hai thứ khác nhau.** `vcd` là VCD thật, dùng
  quy tắc **trừ** của chính nó, và nó xuống **dưới** baseline ở cả hai bảng.
  `vcd_fox_fusion` là ảnh nhiễu của VCD đặt vào quy tắc **cộng** của Fox, và
  được +2.09. Cùng một nhánh âm, hai quy tắc hợp nhất, lệch nhau 3 điểm — nên
  bất kỳ con số "VCD" nào cũng phải nói rõ nó dùng quy tắc nào. Thêm nữa, code
  đã phát hành của Fox **không chạy nổi** VCD và ICD của chính nó (cả hai cờ
  đều dẫn vào nhánh chết), nên không xác định được số trong paper từ đâu ra.

- **Độ dài sinh văn bản quyết định OPERA.** Cùng một implementation: 1 token
  thì +0.07, 10 token thì +2.10. Beam search chấm **cả chuỗi**, nên token đầu
  phụ thuộc có bao nhiêu token theo sau. Fox chạy lại các phương pháp so sánh
  trong harness 1-token của nó, nên ô OPERA trong Table 1 của Fox là hàng
  1-token — hàng mà OPERA gần như không làm gì.

## 7. Quy trình làm việc

GPU ở trên Kaggle, kết quả đi về repo qua git. Có **hai** lệnh đều tên `ingest`
và chúng chạy **ngược chiều nhau** — bỏ lệnh nào cũng mất việc:

```bash
# TRÊN KAGGLE, trước khi chạy: nạp archive đã commit vào runs dir của session
python -m foxrepro.bench.plan ingest results/runs
# (run_kaggle_fox.sh đã gọi sẵn bên trong)
bash run_kaggle_fox.sh <N>
```

```bash
# Ở LOCAL, sau session: thả zip vào runs_inbox/ rồi
python -m foxrepro.bench.plan ingest     # đọc runs_inbox/
python -m foxrepro.bench.plan archive    # ghi vào results/runs/ để commit
python scripts/check_validation.py       # chấm V1–V6
python -m foxrepro.bench.plan plan --budget-hours 10 --round r<N+1>
git commit && git push
```

Ba quy tắc rút ra từ việc mất GPU-h thật:

1. **Push ngay sau khi commit.** Kaggle clone từ remote, nên một commit chưa
   push là một commit không tồn tại. Đã từng chạy 4 session bằng code cũ vì
   quên push.
2. **Ingest cả zip của session bị crash.** Nó thường chứa những job đã xong
   trước lúc crash.
3. **Tăng số round mỗi lần đóng băng plan lại.** Nếu không, session 1 mới sẽ
   ghi đè zip cũ cùng tên.

## 8. Còn lại gì — việc có thể nhận

Plan hiện tại đã cạn. Những mục dưới đây **chưa** nằm trong plan; đánh giá chi
phí lấy từ [`docs/PAPER_AUDIT.md`](PAPER_AUDIT.md).

| việc | GPU-h | cần code mới? | giá trị |
|---|---|---|---|
| **Table 13 + Figure 3, 7, 1a** — chẩn đoán | ~1 | có: một pass forward-only, không sinh text | cao: paper khai báo ROC-AUC **đúng 1.00**, đáng kiểm tra trực tiếp; đồng thời là input cho quy tắc tự động của Table 15 |
| **Table 14 + Figure 19** — quy tắc hợp nhất | ~5 | ít: hai dòng trong `reference_fuse` | cao: arm `off` cho thấy riêng phần hợp nhất mang +2.21 trong +5.03 của Fox; mục này tách quy tắc gating bên trong nó |
| **Table 4 / 12** — vị trí truy vấn | ~17 (hoặc ~5 nếu chỉ POPE-adv) | có: bộ chọn anchor | trung bình-cao: code gốc dùng `tail_n = 1`, một vị trí, **không phải** hai anchor như paper mô tả. Mục này đo cái giá của lựa chọn đó |
| **MME harness** | ~4 | có, viết từ đầu | trung bình: release không có script MME |
| **MMBench** | ~6 | có (SID có script tham khảo) | trung bình |
| **MathVista** | ~8 | có, không repo nào có script | thấp-trung bình |
| **Backbone mới** (Qwen3-VL-8B, InternVL3-8B) | ~20 | có: tổng quát hoá loader; `locate_image_span` đang hard-code 576 token | cao nhưng đắt |
| **GPT-4V judge** | 0 GPU, tốn API | có | đang hoãn |

Ngoài ra: **`docs/PAPER_AUDIT.md` đang lỗi thời** — vẫn ghi SID/CausalMM là
BLOCKED (giờ đã implement xong) và đánh giá thấp tính khả thi của MME/MMBench.
Cần viết lại. Việc này không tốn GPU.

## 9. Bản đồ tài liệu

Đọc theo thứ tự này:

| file | dùng khi |
|---|---|
| **`docs/CS2229_Overview.md`** | file này — tổng quan, đọc trước |
| [`docs/CATCHUP.md`](CATCHUP.md) | **từ điển thuật ngữ đầy đủ** (POPE, CHAIR, ULP, TOST, Holm…) + lịch sử. Đọc khi gặp từ không hiểu. Một số bảng số liệu trong đó cũ hơn file này |
| [`docs/VALIDATION.md`](VALIDATION.md) | 6 dự đoán đăng ký trước và kết quả. Đọc trước khi tin bất kỳ con số nào của một phương pháp so sánh |
| [`docs/DEVIATIONS.md`](DEVIATIONS.md) | mọi chỗ code của ta lệch khỏi release, và vì sao. Dài nhất, tra cứu theo mục |
| [`docs/PAPER_AUDIT.md`](PAPER_AUDIT.md) | kiểm kê 15 bảng + 22 hình: cái nào chạy được, giá bao nhiêu (đang lỗi thời) |
| [`README.md`](../README.md) | quickstart và layout repo |

Bản đồ code:

```
src/foxrepro/
  fox/reference.py    chép nguyên văn từ release — oracle số học
  fox/kernel.py       port sạch, test khẳng định trùng bit với trên
  fox/attention.py    FoxState, đăng ký attention, vòng giải mã hai nhánh
  fox/opera.py        beam search của OPERA (không có rollback — xem DEVIATIONS)
  bench/methods.py    mỗi phương pháp là một class, kèm các control arm
  bench/plan.py       ma trận mục tiêu, theo dõi tiến độ, chia session theo budget
  bench/generate.py   CLI chạy một job
  bench/evaluate.py   POPE/CHAIR → metric
  bench/report.py     bảng Markdown kèm khoảng tin cậy
  bench/stats.py      TOST, Holm, bootstrap ghép đôi
scripts/check_validation.py   chấm V1–V6 (CPU, không cần GPU)
```

## 10. Cạm bẫy — đừng lặp lại

Mỗi dòng dưới đây đã tốn thời gian thật và giờ đều có test chặn.

| Lỗi | Hậu quả |
|---|---|
| Nạp checkpoint chưa convert | vision tower khởi tạo **ngẫu nhiên**; caption trôi chảy nhưng không liên quan ảnh, **không metric nào bắt được** |
| Tự viết prompt | baseline POPE cao hơn paper 5.13 điểm |
| Kaggle chưa gắn GPU | **20.8 h mất trắng**, 724 s/caption thay vì 7.8 s |
| Đọc CHAIR_S mà không xem recall | `bf16` nhìn như thắng 23.6 điểm, thật ra recall tụt 79→60 |
| Dùng p-value cao làm bằng chứng tương đương | luận điểm trung tâm dựa trên nền yếu → giờ dùng **TOST** |
| Không hiệu chỉnh đa so sánh | tự mắc đúng lỗi đang phê phán paper → giờ dùng **Holm** |
| `hash()` của Python cho seed theo sample | `hash()` trên `str` bị salt theo tiến trình, nên cùng một lệnh không tái lập được token của chính nó → giờ dùng `zlib.crc32` |
| Quên `@torch.inference_mode()` | graph autograd giữ KV buffer; job 10-token OOM ngay sample đầu |
| Đo token-identity dưới **sampling** | một lần rút multinomial lật bất cứ khi nào logit nhích, nên hai arm có cùng phân phối vẫn khác nhau từng token. Test identity **phải** chạy greedy |

Hai dòng cuối là bài học tổng quát nhất: **một thí nghiệm có thể trả lời sai
câu hỏi mà vẫn chạy trót lọt.** Khi thiết kế arm mới, hỏi trước: nếu cơ chế này
hoàn toàn không hoạt động, arm này có cho ra kết quả khác không?
