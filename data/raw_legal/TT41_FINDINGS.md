# TT 41/2016/TT-NHNN — source verification and provision mapping for Q029–Q038

Investigated 2026-09-11. Scope: the 10 Basel II capital-adequacy questions that all
cite `https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/7/22-nhnn.pdf` (= **22/2024/TT-NHNN**,
which amends TT 50/2018 on licensing paperwork and contains no capital-adequacy content).

Correct instrument confirmed: **Thông tư số 41/2016/TT-NHNN** ngày 30/12/2016,
"Quy định tỷ lệ an toàn vốn đối với ngân hàng, chi nhánh ngân hàng nước ngoài",
Công báo số 77 + 78 ngày 22-01-2017, hiệu lực 01/01/2020.

## 1. Source that was downloaded

Saved to `data/raw_legal/41_2016_TT_NHNN.pdf` (5,109,560 bytes, 74 pages, sha1 `7da38a286371a45c…`).

Two byte-identical URLs (verified same sha1, same byte count):

| URL | HTTP | bytes | note |
|---|---|---|---|
| `https://congbaocdn.chinhphu.vn/CongBaoCP/VanBan/2016/12/21991/16370-1-412016tt-nhnn16525pdf` | 200 | 5,109,560 | canonical, **use this one** |
| `https://g7.cdnchinhphu.vn/api/download/stream?Url=tm-8mq…&file_name=41_2016_TT-NHNN(16525).pdf` | 200 | 5,109,560 | same file; host has a broken TLS chain (needs `verify=False`) |

Landing page (human-verifiable, not a soft-404): `https://congbao.chinhphu.vn/van-ban/thong-tu-so-41-2016-tt-nhnn-21991.htm`
— note the id is **21991**, not 14181. It states Ban hành 30/12/2016, Hiệu lực 01/01/2020, Công báo 77 + 78.
The attachment URL was mined out of that page's HTML (`data-file="41_2016_TT-NHNN(16525).pdf"`).

Also reachable: `https://vanban.chinhphu.vn/?pageid=27160&docid=188256` (200) — but its only
attachment is `https://datafiles.chinhphu.vn/cpp/files/vbpq/2017/02/41-nhnn.signed.pdf`,
which is a **scan whose text layer is destroyed**: 74 pages, viet_ratio **0.0000**, 0 hits for
`Điều`. Do not ingest that one.

### Diacritic quality measurement

Metric definition was calibrated against a value the project already recorded, so the numbers
are comparable to the existing manifest entries. Calibration anchor:
`data/raw_legal/ocr/22-2023-tt-nhnn.ocr.txt` states `ocr_viet_ratio: 0.2937`; recomputing on
its body gives **0.2941** with the definition *"count of letters with `ord(c) > 127` ÷ count of
all Unicode letters"*. (A narrower definition counting only Vietnamese-unique glyphs gives
0.2381 on the same file, so the project uses the broad one.)

| file | extractor | letters | viet_ratio | verdict |
|---|---|---|---|---|
| `41_2016_TT_NHNN.pdf` (congbao) | pdfplumber | 99,856 | **0.2792** | clean, inside 0.26–0.30 |
| `41_2016_TT_NHNN.pdf` (congbao) | pypdf | 99,844 | 0.2792 | clean but inserts spurious intra-word spaces |
| `41-nhnn.signed.pdf` (datafiles) | pypdf | 98,009 | **0.0000** | scan / diacritic-stripped — unusable |
| Phụ lục 1 region (pages 31–44) | rotated-char reconstruction | — | **0.3045** | clean once de-rotated |

**Use pdfplumber, not pypdf.** pypdf yields `t ỷ l ệ an toàn v ốn` — diacritics are present but
spurious spaces land inside words, which will break exact-substring matching in a retrieval
pipeline. pdfplumber yields `tỷ lệ an toàn vốn`.

### Two extraction traps in this file

1. **Phụ lục 1 (pages 31–44, 0-indexed) is landscape/rotated.** `page.extract_text()` returns
   ~1,300–1,950 chars per page of *reversed single characters*
   (`CỤL ỤHP ÓC ỰT NỐV HNÍT ỂĐ…` = "PHỤ LỤC … ĐỊNH NGHĨA VỀ VỐN"). It is not empty, so a
   pipeline will silently index garbage. The char matrix is `(0, 14.04, -14.04, 0, e, f)`, so
   lines run in −x and reading order is −top. Reconstruction that works:

   ```python
   from collections import defaultdict
   def recon_rotated(page, tol=5):
       ch = [c for c in page.chars if not c.get("upright", True)]
       g = defaultdict(list)
       for c in ch:
           g[round(c["x0"] / tol)].append(c)
       return "\n".join(
           "".join(c["text"] for c in sorted(g[k], key=lambda c: c["top"], reverse=True))
           for k in sorted(g)
       )
   ```

   Phụ lục 1 is where **vốn tự có / Vốn cấp 1 / Vốn cấp 2 and the Tier-2 cap live**, i.e. it
   is load-bearing for Q030 and Q036. If the ingest path does not handle rotation, those two
   questions become unanswerable from the corpus and will generate false hallucination labels.

2. **Phụ lục 6 (asset-classification guidance) has no text layer.** Page 72 carries only the
   heading (267 chars) plus 2 embedded images; page 73 has 0 chars. Only the heading is
   recoverable. Low impact — Q034 is anchored by the residual clause Điều 9.18, not by PL6.

Page-by-page audit: pages 0–30 upright and clean (Điều 6 → p10, Điều 9 → p13, Điều 16 → p25,
Điều 19 → p29, Điều 24 → p30); pages 31–44 rotated (Phụ lục 1); pages 45–72 upright and clean
(PL2 → p45, PL3 → p50, PL4 → p52, PL5 → p70, PL6 heading → p72); page 73 empty.

## 2. Real structure of TT 41/2016 (so the claimed numbers can be checked)

Điều 1 Phạm vi · 2 Giải thích từ ngữ · 3 Cơ cấu tổ chức, kiểm toán nội bộ · 4 Dữ liệu, CNTT ·
5 Doanh nghiệp xếp hạng tín nhiệm độc lập · **6 Tỷ lệ an toàn vốn** · **7 Vốn tự có** ·
**8 Tài sản tính theo rủi ro tín dụng** · **9 Hệ số rủi ro tín dụng (CRW)** · 10 Hệ số chuyển
đổi (CCF) · 11 Giảm thiểu rủi ro tín dụng · 12–15 credit-risk mitigation ·
**16 Vốn yêu cầu cho rủi ro hoạt động** · 17 trạng thái rủi ro · **18 Vốn yêu cầu cho rủi ro
thị trường** · **19 Chế độ báo cáo** · 20 Công bố thông tin · 21–22 trách nhiệm NHNN ·
23 Hiệu lực · 24 Tổ chức thực hiện. Phụ lục 1 vốn tự có · 2 RWACCR · 3 chỉ số kinh doanh ·
4 rủi ro thị trường · 5 công bố thông tin · 6 phân loại tài sản.

The gold passages claim Điều 9, 5, 8, 6, 14, 11. **Điều 14 is not a capital rule at all** (it is
"Giảm thiểu rủi ro tín dụng bằng bảo lãnh của bên thứ ba"). Every claimed number except the
bare "Phụ lục" ones is wrong.

## 3. Provision-by-provision mapping

All quotes are verbatim from `data/raw_legal/41_2016_TT_NHNN.pdf` via pdfplumber, except where
marked **[TT 22/2023]** (from the amending circular) or **[TT 35/2015]**.

### Q029 — minimum CAR = 8%
- claimed: `Điều 9` → **real: Điều 6 khoản 2** (and khoản 3 for banks with subsidiaries)
- > "2. Ngân hàng không có công ty con, chi nhánh ngân hàng nước ngoài phải thường xuyên duy trì tỷ lệ an toàn vốn xác định trên cơ sở báo cáo tài chính của ngân hàng, chi nhánh ngân hàng nước ngoài tối thiểu 8%."
- > "3. Ngân hàng có công ty con phải duy trì: a) Tỷ lệ an toàn vốn xác định trên cơ sở báo cáo tài chính của ngân hàng tối thiểu 8%; b) Tỷ lệ an toàn vốn hợp nhất xác định trên cơ sở báo cáo tài chính hợp nhất của ngân hàng tối thiểu 8%."
- Gold answer is substantively **correct**; only the locator is wrong.

### Q030 — vốn tự có = vốn cấp 1 + vốn cấp 2
- claimed: `Điều 5` → **real: Điều 7 khoản 2**, detail in **Phụ lục 1, Phần A, mục I**
- > "2. Vốn tự có bao gồm tổng Vốn cấp 1 và Vốn cấp 2 trừ đi các khoản giảm trừ quy định tại Phụ lục 1 ban hành kèm theo Thông tư này."
- Phụ lục 1 (rotated pages) enumerates: `VỐN CẤP 1 (A) = A1 - A2`, `Cấu phần Vốn cấp 1 (A1) = ∑1 ÷ 7` = (1) Vốn điều lệ, (2) Quỹ dự trữ bổ sung vốn điều lệ, (3) Quỹ đầu tư phát triển nghiệp vụ, (4) Quỹ dự phòng tài chính, (5) Vốn đầu tư xây dựng cơ bản, mua sắm tài sản cố định, (6) Lợi nhuận chưa phân phối, (7) Thặng dư vốn cổ phần. Deductions `A2 = ∑8 ÷ 10` = (8) Lợi thế thương mại, (9) Lỗ lũy kế, (10) Cổ phiếu quỹ.
- Gold answer ("vốn điều lệ, quỹ dự trữ, lợi nhuận không chia" / "trái phiếu chuyển đổi, nợ thứ cấp, dự phòng chung") is **correct but incomplete** — it omits the deduction items (`trừ đi các khoản giảm trừ`).

### Q031 — risk weight 0%
- claimed: `Phụ lục` → **real: Điều 9 khoản 2 and khoản 3** (not an annex)
- > "2. Đối với tài sản là tiền mặt, vàng và các khoản tương đương tiền mặt của ngân hàng, chi nhánh ngân hàng nước ngoài, hệ số rủi ro tín dụng là 0%."
- > "3. Đối với tài sản là khoản phải đòi Chính phủ Việt Nam, Ngân hàng Nhà nước, Kho bạc Nhà nước, Ủy ban nhân dân tỉnh, thành phố trực thuộc Trung ương, các ngân hàng chính sách, hệ số rủi ro tín dụng là 0%. Đối với khoản phải đòi Công ty Quản lý tài sản của các tổ chức tín dụng Việt Nam (VAMC), Công ty trách nhiệm hữu hạn Mua bán nợ Việt Nam (DATC), hệ số rủi ro là 20%."
- Gold answer is **correct**. "Tiền gửi tại NHNN" maps to khoản 3 (khoản phải đòi Ngân hàng Nhà nước); "trái phiếu Chính phủ" maps to khoản 3 (khoản phải đòi Chính phủ Việt Nam). Also 0% for khoản phải đòi tổ chức tài chính quốc tế (khoản 4).

### Q032 — residential-mortgage risk weight "50% when LTV ≤ 70%" ⚠️
- claimed: `Phụ lục` → **real: Điều 9 khoản 11 điểm b**, as amended by **[TT 22/2023] khoản 8 Điều 1**
- **There is no 70% LTV threshold anywhere in TT 41/2016 or TT 22/2023.** The actual grid is
  two-dimensional (LTV × DSC, where DSC = "Tỷ lệ thu nhập" = annual debt service / annual income):

  `Điều 9.11.b` (unchanged by TT 22/2023, which renumbered it as b(ii)):
  | LTV | <40% | 40–<60% | 60–<80% | 80–<90% | 90–<100% | ≥100% |
  |---|---|---|---|---|---|---|
  | DSC ≤ 35% | 25% | 30% | 40% | **50%** | 60% | 80% |
  | DSC > 35% | 30% | 40% | **50%** | 70% | 80% | 100% |

  **[TT 22/2023]** added b(i) for social housing / Government-supported programmes:
  | LTV | <40% | 40–<60% | 60–<80% | 80–<90% | 90–<100% | ≥100% |
  |---|---|---|---|---|---|---|
  | DSC ≤ 35% | 20% | 25% | 30% | 35% | 40% | 45% |
  | DSC > 35% | 25% | 30% | 35% | 40% | 45% | 50% |

- So a 50% weight attaches at **LTV 60%–<80% with DSC ≤ 35%**, or **LTV 80%–<90% with DSC > 35%** — never at "LTV ≤ 70%".
- The related non-business real-estate grid (`Điều 9.10.b`, identical before and after TT 22/2023) is LTV-only: <40% → 30%, 40–<60% → 40%, **60–<80% → 50%**, 80–<90% → 70%, 90–<100% → 80%, ≥100% → 100%. If the intended question was "real-estate-secured exposure at 50%", the correct condition is **LTV 60%–<80%**, not ≤70%.
- Verbatim, Điều 9.11.b as originally enacted: > "b) Hệ số rủi ro áp dụng cho khoản cho vay thế chấp nhà ở theo Tỷ lệ bảo đảm (LTV) và Tỷ lệ thu nhập (DSC) như sau:"
- Verbatim, **[TT 22/2023]** khoản 1 Điều 1 (this is what actually changed): > "1. Sửa đổi, bổ sung khoản 11 Điều 2 như sau: '11. Khoản cho vay thế chấp nhà là khoản cho vay bảo đảm bằng bất động sản đối với cá nhân để mua nhà, bao gồm: a) Khoản cho vay bảo đảm bằng bất động sản đối với cá nhân để mua nhà đáp ứng các điều kiện sau: i) Nguồn tiền trả nợ không phải là nguồn tiền cho thuê nhà hình thành từ khoản cho vay; ii) Nhà đã được hoàn thành để bàn giao theo hợp đồng mua bán nhà; iii) Ngân hàng, chi nhánh ngân hàng nước ngoài có đầy đủ quyền hợp pháp để xử lý nhà thế chấp…; iv) Nhà hình thành từ khoản cho vay thế chấp này phải được định giá độc lập… b) Khoản cho vay để mua nhà ở xã hội, mua nhà ở theo các chương trình, dự án hỗ trợ của Chính phủ…'"
- **This question needs rewriting, not just re-locating.** As written its gold answer is not
  supported by any version of the instrument.

### Q033 — RWA = credit + operational + market ⚠️
- claimed: `Điều 8` → **real: Điều 6 khoản 1** (the aggregation is in the CAR denominator)
- > "1. Tỷ lệ an toàn vốn (CAR) tính theo đơn vị phần trăm (%) được xác định bằng công thức: CAR = C / (RWA + 12,5 x (KOR + KMR)) x 100%" — "Trong đó: - C: Vốn tự có; - RWA: Tổng tài sản tính theo rủi ro tín dụng; - KOR: Vốn yêu cầu cho rủi ro hoạt động; - KMR: Vốn yêu cầu cho rủi ro thị trường."
- **Điều 8 does NOT say RWA = credit + operational + market.** Điều 8.1 says: > "Tổng tài sản tính theo rủi ro tín dụng (RWA) bao gồm tổng tài sản tính theo rủi ro tín dụng (RWACR) và tổng tài sản tính theo rủi ro tín dụng đối tác (RWACCR) được tính theo công thức: RWA = RWACR + RWACCR". In TT 41/2016 the symbol RWA denotes *credit* risk only; operational and market risk enter as capital charges KOR and KMR scaled by 12.5.
- The gold answer's *substance* (three risk types drive the capital requirement) is right, but the formula it attributes to Điều 8 is wrong, and citing Điều 8 as the supporting passage would contradict the answer.

### Q034 — risk weight 100% ⚠️
- claimed: `Phụ lục` → **real: Điều 9 khoản 18** for the residual-asset 100%; the corporate part is **not** a flat 100%
- > "18. Đối với các tài sản khác trên bảng cân đối kế toán trừ các tài sản quy định tại khoản 1, khoản 2, khoản 3, khoản 4, khoản 5, khoản 6, khoản 7, khoản 8, khoản 9, khoản 10, khoản 11, khoản 12, khoản 13, khoản 14, khoản 15, khoản 16 và khoản 17 Điều này, hệ số rủi ro tín dụng là 100%."
- "Tài sản cố định" and "bất động sản đầu tư" appear **nowhere** in TT 41/2016's risk-weight provisions (the only near-match is the Điều 2.13 definition of "Bất động sản kinh doanh"). They reach 100% only through the residual clause above.
- "Cho vay doanh nghiệp thông thường" is **not** 100%. Điều 9.9.b sets a matrix on revenue × leverage × equity:
  | | doanh thu <100 tỷ | 100–<400 tỷ | 400–1500 tỷ | >1500 tỷ |
  |---|---|---|---|---|
  | đòn bẩy <25% | 100% | 80% | 60% | 50% |
  | đòn bẩy 25–50% | 125% | 110% | 95% | 80% |
  | đòn bẩy >50% | 160% | 150% | 140% | 120% |
  | vốn CSH âm hoặc = 0 | 250% | 250% | 250% | 250% |
  Plus Điều 9.9.a: SMEs → 90%; 9.b(ii): no financial statements provided → 200%; 9.b(iii): enterprises operating <1 year → 150%.
- 100% is the correct weight only in the *specific* cell "doanh thu dưới 100 tỷ đồng và tỷ lệ đòn bẩy dưới 25%". The gold answer's blanket "cho vay doanh nghiệp thông thường = 100%" is not supported.

### Q035 — risk weight 150% ⚠️
- claimed: `Phụ lục` → **real: Điều 9 khoản 13 điểm a** for the NPL limb; the real-estate-business limb is **wrong**
- > "13. Đối với khoản nợ xấu, hệ số rủi ro tín dụng áp dụng như sau: a) Đối với khoản nợ xấu có dự phòng cụ thể nhỏ hơn 20% giá trị của khoản nợ xấu (trừ khoản nợ xấu là khoản cho vay thế chấp nhà có dự phòng cụ thể nhỏ hơn 20% giá trị của khoản nợ xấu), hệ số rủi ro tín dụng là 150%;"
- TT 41/2016 uses **"nợ xấu"**, never "nợ quá hạn trên 90 ngày". That older phrasing belongs to the pre-Basel-II regime, not to this circular.
- Real-estate **business** lending is not 150%:
  - Điều 9.10.c (LTV grid for BĐS kinh doanh): LTV <60% → **75%**, 60–<75% → **100%**, ≥75% → **120%** (unchanged by TT 22/2023)
  - Điều 9.10.đ: **150%** where the bank has no LTV information for a real-estate-secured exposure
  - Điều 9.10.e, as amended by **[TT 22/2023]**: > "e) Hệ số rủi ro tín dụng 200% được áp dụng đối với tài sản là khoản cấp tín dụng chuyên biệt dưới hình thức cấp tín dụng tài trợ dự án kinh doanh bất động sản. Trường hợp đối với tài sản là khoản cấp tín dụng chuyên biệt dưới hình thức cấp tín dụng tài trợ dự án kinh doanh bất động sản khu công nghiệp, hệ số rủi ro tín dụng là 160%." (the 2016 original said simply "200% ... đối với tài sản là khoản cấp tín dụng tài trợ dự án kinh doanh bất động sản")
- Other genuine 150% cases in Điều 9: unrated/below-B- foreign sovereigns (khoản 5), enterprises operating <1 year (khoản 9.b(iii)), equity holdings and loans for securities investment/margin lending (khoản 15).
- **This question needs rewriting.** Only the NPL limb survives, and even that needs the
  "dự phòng cụ thể nhỏ hơn 20%" qualifier to be true.

### Q036 — Tier 2 capped at 100% of Tier 1
- claimed: `Điều 6` → **real: Phụ lục 1, Phần A, mục I** (and mục III for consolidated, mục for foreign-bank branches)
- > "VỐN CẤP 2 (B) = B1 - B2 - 20    Giá trị vốn cấp 2 tối đa bằng vốn cấp 1"
- Consolidated: > "Giá trị vốn cấp 2 hợp nhất tối đa bằng vốn [cấp 1 hợp nhất]" / "VỐN CẤP 2 HỢP NHẤT (B) = B1 - B2 - 22"
- Foreign bank branches: > "VỐN CẤP 2 (B) = B1 - B2 - (13)    Giá trị vốn cấp 2 tối đa bằng vốn cấp 1."
- Gold answer is **correct**; the locator is wrong, and the text lives on the **rotated** pages — see trap 1 above.

### Q037 — CAR reporting frequency to NHNN ⚠️ **not in this instrument at all**
- claimed: `Điều 14` → TT 41/2016's only reporting provision is **Điều 19**, and it states **no frequency**:
- > "Điều 19. Chế độ báo cáo — Ngân hàng, chi nhánh ngân hàng nước ngoài thực hiện báo cáo tỷ lệ an toàn vốn theo quy định của Ngân hàng Nhà nước về chế độ báo cáo thống kê đối với tổ chức tín dụng, chi nhánh ngân hàng nước ngoài."
- (Điều 14 is "Giảm thiểu rủi ro tín dụng bằng bảo lãnh của bên thứ ba" — unrelated. Điều 20 is *public disclosure*, "Định kỳ 6 tháng một lần theo năm tài chính", which is not NHNN reporting.)
- The delegated instrument is **Thông tư 35/2015/TT-NHNN** ngày 31/12/2015, "Quy định Chế độ
  báo cáo thống kê áp dụng đối với các tổ chức tín dụng, chi nhánh ngân hàng nước ngoài"
  (amended by **TT 11/2018/TT-NHNN**). Its reporting schedule, verbatim:
  - > "10 Báo cáo tình hình thực hiện tỷ lệ an toàn vốn tối thiểu | 118-TTGS | Tháng | 12 hàng tháng | 308"
  - > "11 Báo cáo tài sản có rủi ro riêng lẻ | 119.1-TTGS | Tháng | 12 hàng tháng | 309"
  - > "12 Báo cáo tài sản có rủi ro hợp nhất | 119.2-TTGS | Quý | 18 của tháng đầu quý tiếp theo | 313"
  - > "13 Báo cáo vốn tự có riêng lẻ | 120.1-TTGS | Tháng | 12 hàng tháng | 317"
  - > "14 Báo cáo vốn tự có hợp nhất | 120.2-TTGS | Quý | 18 của tháng đầu quý tiếp theo | 321"
  - > "15 Báo cáo vốn tự có của chi nhánh ngân hàng nước ngoài | 120.3-TTGS | Tháng | 12 hàng tháng | 325"
- So the gold answer "hàng tháng và hàng quý" is **substantively correct**, but **no capital
  circular in this corpus contains it**. Q037 will produce a false hallucination label unless
  TT 35/2015 (as amended) is added to the corpus or the question is dropped.

### Q038 — operational-risk capital = 15% × 3-year average ⚠️
- claimed: `Điều 11` → **real: Điều 16 khoản 1** (Điều 11 is "Giảm thiểu rủi ro tín dụng")
- > "1. Vốn yêu cầu cho rủi ro hoạt động (KOR) được xác định bằng công thức: KOR = (BInăm thứ n + BInăm thứ n-1 + BInăm thứ n-2) / 3 x 15%" — "Trong đó: - BInăm thứ n: Chỉ số kinh doanh được xác định theo quý gần nhất tại thời điểm tính toán; - BInăm thứ n-1, BInăm thứ n-2: Chỉ số kinh doanh được xác định theo quý tương ứng của 2 năm liền kề trước năm tính toán."
- > "2. Chỉ số kinh doanh được xác định theo công thức sau: BI = IC + SC + FC" — IC = |thu nhập lãi và tương tự − chi phí lãi và tương tự|; SC = dịch vụ + hoạt động khác; FC = |lãi/lỗ thuần từ kinh doanh ngoại hối, mua bán chứng khoán kinh doanh và đầu tư|. Detail in Phụ lục 3.
- Two precision problems with the gold answer:
  1. The base is **"Chỉ số kinh doanh" (Business Indicator)**, not "thu nhập". BI is a
     three-component construct, and it is measured **per quarter** ("theo quý gần nhất" and the
     corresponding quarters of the two preceding years) — not as three annual income figures.
  2. It is **not** the "Basic Indicator Approach (BIA)". BIA is the Basel II Pillar-1 option
     based on gross income; TT 41/2016 implements the Basel III **Business Indicator**
     formulation. Labelling it BIA is a factual error that a knowledgeable model could
     correctly contradict — which would then be scored as a hallucination.
- The "15% × 3-year average" arithmetic is right; the naming and the base are not.

## 4. Amendment status and ingest recommendation

Chain, verified from primary/official sources:

| instrument | date | effect on TT 41/2016 |
|---|---|---|
| **TT 41/2016/TT-NHNN** | issued 30/12/2016, effective 01/01/2020 | the base text |
| **TT 26/2022/TT-NHNN** | 31/12/2022 | its **khoản 2 Điều 1** replaces **Điều 23** (hiệu lực thi hành / transition registration) of TT 41/2016. **No substantive capital rule is touched.** TT 26/2022's main target is TT 22/2019 (prudential limits). |
| **TT 22/2023/TT-NHNN** | issued 29/12/2023, **effective 01/07/2024** | **material**. 17 clauses amending: khoản 11 Điều 2; điểm c khoản 12 Điều 2; khoản 15 Điều 2; khoản 3 Điều 8; khoản 7 Điều 9; điểm b khoản 9 Điều 9; **khoản 10 Điều 9**; **điểm b khoản 11 Điều 9**; new khoản 12a Điều 9 (rural/agricultural personal loans → 50%); điểm e khoản 3 Điều 11; khoản 4 Điều 11; Điều 12; Điều 17; khoản 4 Điều 18; khoản 1 Điều 21; khoản 2 Điều 22 |
| **TT 14/2025/TT-NHNN** | issued 30/06/2025, effective **15/09/2025** (per its Điều 82) | replaces the regime, but with a transition: **TT 41/2016 + TT 26/2022 + TT 22/2023 remain applicable until 31/12/2029** for banks that have not registered for the Standardized Approach and have not been approved for IRB. **All three are repealed on 01/01/2030.** |

Luật Các tổ chức tín dụng 2024 (32/2024/QH15) does **not** itself set the CAR regime — it
authorises NHNN to regulate capital adequacy, and TT 14/2025's own legal basis cites it
("Pursuant to the Law on Credit Institutions No. 32/2024/QH15…"). No decree was found that
supersedes the CAR regime ahead of TT 14/2025.

**Status as of 2026-09-11: TT 41/2016 (as amended by TT 26/2022 and TT 22/2023) IS STILL IN
FORCE.** TT 14/2025 is in force in parallel but optional until 01/01/2030.

### Recommendation

**Ingest TT 41/2016 (`data/raw_legal/41_2016_TT_NHNN.pdf`) PLUS TT 22/2023. Do not ingest the
2016 original alone, and do not ingest TT 14/2025 for these questions.**

Reasoning:

1. **TT 41/2016 alone is stale for Q032/Q034/Q035.** TT 22/2023 rewrites khoản 10 Điều 9 and
   điểm b khoản 11 Điều 9 — precisely the real-estate and residential-mortgage risk-weight
   tables those three questions are about. This is the same failure mode the manifest already
   recorded for the 39/2016 → 06/2023 pair.
2. **TT 14/2025 is the wrong target for a benchmark whose gold answers describe the 41/2016
   regime.** It keeps 8% CAR but adds capital buffers, reclassifies real estate exposures on
   additional criteria beyond LTV/DSC, and is not mandatory for all banks until 01/01/2030.
   Gold answers written against the LTV/DSC grids would not be supported by it.
3. **TT 26/2022 is optional.** It only replaces Điều 23 (transition/registration mechanics),
   which none of the 10 questions touches. Include it only if the corpus is meant to be
   complete; it changes no gold label.
4. **Q037 cannot be fixed by any choice of capital circular** — it needs TT 35/2015/TT-NHNN
   (as amended by TT 11/2018/TT-NHNN) in the corpus, or the question must be dropped or
   re-pointed.

### Practical notes for whoever applies this

- TT 22/2023's authoritative PDF is
  `https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/01/22-nhnn.pdf` (200, 23,767,020 bytes) —
  confirmed by its page-1 header `Số: 22/2023/TT-NHNN … ngày 29 tháng 12 năm 2023 … Sửa đổi,
  bổ sung một số điều của Thông tư số 41/2016/TT-NHNN`. It is a **scan with a corrupt text
  layer**, so it needs OCR. The `DOC_MANIFEST.json` entry for this URL is correct.
- **`data/raw_legal/22-2023-tt-nhnn.pdf` is mis-named, and so is its OCR.** Verified at the file
  level, not just from the OCR output: the local PDF is 3,172,097 bytes, sha1 `95d418b05ca72a32`,
  **10 pages**, 0 occurrences of `41/2016`, 5 occurrences of `50/2018`, and its header (read
  through a diacritic-stripped layer) is `Số: …/2024/TT-NHNN … Hà Nội, ngày … tháng 06 năm 2024
  … THÔNG TƯ Sửa đổi, bổ sung một số điều của Thông tư số 50/2018/`. That is **TT 22/2024**, the
  licensing circular — the same wrong document the 10 questions cite. TT 22/2023 is 52 pages and
  amends TT 41/2016, so it is unambiguous. Consequences: (i) whatever job produced this file
  fetched `2024/7/22-nhnn.pdf` and labelled it 22/2023; (ii) `data/raw_legal/ocr/22-2023-tt-nhnn.ocr.txt`
  inherits the error (its own provenance header names `source_file: 22-2023-tt-nhnn.pdf`);
  (iii) **the real TT 22/2023 is not in the repo at all**; (iv) any coverage or passage-match
  claim built on that file is measuring the wrong instrument. This PDF also needs OCR — its text
  layer is corrupt in the same way as the TT 41/2016 `signed` scan.
- A consolidated text (văn bản hợp nhất) of TT 41/2016 does exist —
  **05/VBHN-NHNN năm 2024** — but I could not obtain it from a government source (luatvietnam
  gates the download; the `05/VBHN-NHNN` on congbao.chinhphu.vn, id 25728, is a *different*
  2018 consolidation of the TCTD chart-of-accounts decision, since VBHN numbers restart each
  year). If a single-file ingest is preferred over original-plus-amendment, this is the
  document to chase.

## 5. Could not verify / open items

- **CLOSED 2026-09-11 (later session): clauses 7 and 8 are now OFFICIAL-source verified.**
  The real TT 22/2023 was downloaded from `datafiles.chinhphu.vn/cpp/files/vbpq/2024/01/22-nhnn.pdf`
  (23,767,020 bytes, sha1 `50e2ac7fec5e3945e23efd6f64e9ce3389df036d`, 50 pages — this official
  scan has 50 pp, the 52-pp count below was the VNBA mirror) and OCR'd to
  `data/raw_legal/ocr/22_2023_TT_NHNN.ocr.txt` (viet_ratio 0.281). The OCR confirms clause 7
  (khoản 10 Điều 9): non-business LTV grid 30/40/50/70/80/100; business grid 75/100/120
  (c); 150% with no LTV information (đ); 200% / 160% KCN (e). It confirms clause 8
  (điểm b khoản 11 Điều 9): social-housing grid 20–45 / 25–50 (b.i); ordinary residential
  grid (b.ii) DSC ≤ 35% → 25/30/40/50/60/80 and DSC > 35% → 30/40/50/70/80/100. **Note a
  transposition error in §3's Q032 prose above:** per the official grid, 50% attaches at
  LTV 60–<80% with DSC **>** 35%, or LTV 80–<90% with DSC **≤** 35% (§3's own table was
  right; its prose sentence swapped the two). The corrected gold answer for Q032 uses the
  official-OCR orientation.
- **TT 22/2023's full text is corroborated but not from a government source.** The clause list
  and the amended risk-weight tables above were read off
  `https://luatvietnam.vn/tai-chinh/thong-tu-22-2023-tt-nhnn-sua-doi-tt-41-2016-tt-nhnn-ve-ty-le-an-toan-von-ngan-hang-288187-d1.html`
  (HTTP 200, 111,285 visible chars, header confirmed as `Số: 22/2023/TT-NHNN … ngày 29 tháng 12
  năm 2023`), a commercial aggregator. Clause 1 was independently confirmed verbatim by the
  official NHNN journal
  (`https://tapchinganhang.gov.vn/mot-so-sua-doi-bo-sung-quy-dinh-ve-ti-le-an-toan-von-doi-voi-ngan-hang-chi-nhanh-ngan-hang-nuoc-ngoai-1409.html`).
  Clauses 7 and 8 (the khoản 10 / khoản 11.b Điều 9 rewrites) are **single-sourced** — they
  should be re-verified against the OCR of the official PDF before they are written into gold
  labels. A VNBA copy exists
  (`https://s-vnba-cdn.aicms.vn/vnba-media/24/1/9/thong-tu-so-22-2023-tt-nhnn_659cfabaef4bf.pdf`,
  200, 9,561,968 bytes, 52 pages) but is a **pure scan with no text layer** (51 chars
  extracted, viet_ratio 0.0000).
- **TT 35/2015/TT-NHNN was read from a non-government mirror**
  (`https://ngvgroup.vn/wp-content/uploads/2016/01/Thong-tu-35-2015-TT-NHNN.pdf`, 200,
  9,762,953 bytes, 564 pages, born-digital, viet_ratio 0.2774, header confirmed
  `Số: 35/2015/TT-NHNN … ngày 31 tháng 12 năm 2015`). I did **not** check whether TT 11/2018
  or any later circular changed the frequency or renumbered biểu 118/119/120-TTGS. The
  monthly/quarterly split for CAR reporting is corroborated by that schedule but not by the
  current amended text.
- **`vbpl.vn` is unusable from this machine** — both the `vbpq-van-ban-goc.aspx?ItemID=117310`
  and `vbpq-toanvan.aspx?ItemID=164719` pages return HTTP 200 but are JS shells
  ("Đang tải dữ liệu…", 242–367 visible chars, no instrument content). Add to the bot-wall map.
- **`congbao.chinhphu.vn/tim-kiem?q=…` does not work for CLI clients** (returns the chrome with
  85 navigation links and no results). Find congbao document ids via search-engine `site:`
  queries or by mining the `data-file` / `congbaocdn` attributes out of a known van-ban page.
- **Phụ lục 6 of TT 41/2016 is image-only**, so the asset-classification mapping that would
  formally justify routing "tài sản cố định" / "bất động sản đầu tư" to the Điều 9.18 residual
  100% bucket cannot be read from this file.
- I did not check whether any circular **other than** TT 26/2022 and TT 22/2023 amended TT
  41/2016 between 2016 and 2025. The searches I ran surfaced only those two plus TT 14/2025,
  and one industry source described TT 41/2016 as "đã được sửa đổi, bổ sung năm 2022, năm 2023",
  which is consistent — but that is not an exhaustive check against the official amendment
  history (congbao's "Lược đồ"/"Sơ đồ văn bản" view for id 21991 would settle it and is
  JS-rendered).
