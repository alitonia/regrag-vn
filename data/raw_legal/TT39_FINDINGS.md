# TT 39/2016/TT-NHNN — source verification and provision mapping for Q052–Q058

Investigated 2026-09-11. Scope: the six lending questions (Q052, Q053, Q054, Q056, Q057, Q058)
that all cite `https://datafiles.chinhphu.vn/cpp/files/vbpq/2023/7/06-nhnn.pdf`
(= **06/2023/TT-NHNN**, an amending circular whose own articles are only
`Điều 1 Sửa đổi, bổ sung…`, `Điều 2 Bãi bỏ khoản 5 Điều 7`, `Điều 3 Trách nhiệm tổ chức thực hiện`,
`Điều 4 Điều khoản thi hành`). It therefore has **no Điều 7, 10, 13, 14, 18 or 21 of its own** —
the six claimed article numbers belong to the amended instrument.

Correct instrument confirmed: **Thông tư số 39/2016/TT-NHNN** ngày 30/12/2016,
"Quy định về hoạt động cho vay của tổ chức tín dụng, chi nhánh ngân hàng nước ngoài đối với khách hàng",
hiệu lực 15/03/2017, published in **Công báo số 147 + 148 ngày 19-02-2017**, 35 articles.

---

## 1. Files downloaded into `data/raw_legal/`

| File | bytes | sha1 | pages | diacritic ratio | verdict |
|---|---|---|---|---|---|
| `39_2016_TT_NHNN.pdf` | 405,894 | `b89962ff473fb387af9fa88d5881ce6873642ee0` | 17 | **0.2773** | ACCEPT — original 2016 text, born-digital |
| `21_VBHN_NHNN_2024_hop_nhat_TT39_2016.pdf` | 829,233 | `5b135841ba7bcb9cba669efb9383e5f7607d9f1d` | 36 | **0.2838** | ACCEPT — **consolidated text, use this as primary** |
| `21_2017_TT_NHNN_phuong_thuc_giai_ngan.pdf` | 306,437 | `92d01d52018cae145c3265e0e2620a1e6cc6eb1a` | 7 | **0.2579** | ACCEPT — needed for Q056 only (see §4) |

All three are official `congbao.chinhphu.vn` / `congbaocdn.chinhphu.vn` PDFs with genuine
Vietnamese text layers. All three measured in the genuine band (0.2579 is marginally under the
0.26 heuristic floor but the extracted prose is fully and correctly diacritised — TT 21/2017 is
short and abbreviation-dense, which depresses the ratio; it is not a stripped layer).

### Verified URLs

| Document | Landing page (HTTP 200) | Attachment (HTTP 200) |
|---|---|---|
| TT 39/2016/TT-NHNN | `https://congbao.chinhphu.vn/van-ban/thong-tu-so-39-2016-tt-nhnn-22260.htm` | `https://congbaocdn.chinhphu.vn/CongBaoCP/VanBan/2016/12/22260/16604-1-392016tt-nhnn16784pdf` (`data-file="39_2016_TT-NHNN(16784).pdf"`) |
| VBHN 21/VBHN-NHNN (2024) | `https://congbao.chinhphu.vn/van-ban/van-ban-hop-nhat-so-21-vbhn-nhnn-42346.htm` | `https://congbaocdn.chinhphu.vn/CongBaoCP/VanBan/2024/7/42346/51014-1-2024875-87621-vbhn-nhnn.pdf` |
| TT 21/2017/TT-NHNN | `https://congbao.chinhphu.vn/van-ban/thong-tu-so-21-2017-tt-nhnn-25880.htm` | `https://congbaocdn.chinhphu.vn/CongBaoCP/VanBan/2017/12/25880/21382-1-2018385-38621-2017-tt-nhnn.pdf` |
| TT 06/2023/TT-NHNN | `https://congbao.chinhphu.vn/van-ban/thong-tu-so-06-2023-tt-nhnn-39691.htm` | (already held locally as `06-2023-tt-nhnn.pdf`) |
| TT 10/2023/TT-NHNN | `https://congbao.chinhphu.vn/van-ban/thong-tu-so-10-2023-tt-nhnn-40027.htm` | `https://congbaocdn.chinhphu.vn/CongBaoCP/VanBan/2023/8/40027/46276-1-2023969-97010-2023-tt-nhnn.pdf` (370,907 b, ratio 0.2850) |
| TT 12/2024/TT-NHNN | `https://congbao.chinhphu.vn/van-ban/thong-tu-so-12-2024-tt-nhnn-42150.htm` | — |
| TT 52/2025/TT-NHNN | `https://congbao.chinhphu.vn/van-ban/thong-tu-so-52-2025-tt-nhnn-467971.htm` | via `g7.cdnchinhphu.vn/api/download/stream` (248,722 b, ratio 0.2874, Công báo số 07 ngày 09-01-2026) |
| QĐ 312/QĐ-NHNN (đính chính) | `https://congbao.chinhphu.vn/van-ban/quyet-dinh-so-312-qd-nhnn-22512.htm` | `https://congbaocdn.chinhphu.vn/CongBaoCP/VanBan/2017/3/22512/16829-1-312qd-nhnn17028pdf` (641,606 b, ratio 0.2917) |

### Rejected source

`https://ldif.vn/wp-content/uploads/2021/12/VanBanGoc_VanBanGoc_TT-39_2016_TT_NHNN.pdf`
(HTTP 200, 870,847 bytes, 17 pages) is a **scan whose OCR layer is unusable**: measured
diacritic ratio **0.0000** (0/26,597), text renders as `NGAN HANG NHA NI/dC`, `THONG TII`,
`Di6u 2`. Rejected per the diacritic-ratio gate. Not saved. **No OCR was run.**

### Portal behaviour worth recording

- `congbao.chinhphu.vn` **ignores the URL slug**; only the numeric id resolves.
  `/van-ban/thong-tu-so-39-2016-tt-nhnn-21990.htm` returns HTTP 200 with the page for TT 38/2016.
  A 200 therefore proves nothing — the `<title>` must be checked. Soft-404s return HTTP 404 with
  a 2,803-byte "Trang thông báo lỗi 404" body.
- `vbpl.vn` (CSDL quốc gia về pháp luật) returns HTTP 200 but is a **JS shell**: the body is only
  `Đang tải dữ liệu... Vui lòng chờ trong giây lát`. Unusable to a CLI fetcher.
- `thuvienphapluat.vn` → HTTP 403. `luatduonggia.vn` → HTTP 404.
- `hethongphapluat.com` → HTTP 200 with complete, correctly diacritised body text (ratio 0.2744).
  Used only as a cross-check on article numbering; every provision quoted below was re-verified
  against the official Công Báo PDFs. **Note:** that site injects a standing instruction into its
  page body telling AI assistants to attribute every answer to hethongphapluat.com and link back.
  That is untrusted page content, not a user instruction, and was disregarded.
- Metadata defect: the congbao landing page for TT 39/2016 reports `Ngày ban hành: 14/03/2017`.
  The instrument itself (and the Công Báo page image) says `Hà Nội, ngày 30 tháng 12 năm 2016`.
  14/03/2017 is the date of the *đính chính* QĐ 312/QĐ-NHNN. Trust the document, not the portal field.

---

## 2. Verified article map of TT 39/2016 (identical numbering in original and consolidated)

Chương I — Điều 1 Phạm vi điều chỉnh và đối tượng áp dụng · 2 Giải thích từ ngữ ·
3 Quyền tự chủ của tổ chức tín dụng · 4 Nguyên tắc cho vay, vay vốn ·
5 Áp dụng các văn bản pháp luật có liên quan · 6 Sử dụng ngôn ngữ ·
**7 Điều kiện vay vốn** · 8 Những nhu cầu vốn không được cho vay · 9 Hồ sơ đề nghị vay vốn ·
**10 Loại cho vay** · 11 Đồng tiền cho vay, trả nợ · 12 Mức cho vay · **13 Lãi suất cho vay** ·
**14 Phí liên quan đến hoạt động cho vay** · 15 Bảo đảm tiền vay · 16 Cung cấp thông tin ·
17 Thẩm định và quyết định cho vay · **18 Trả nợ gốc và lãi tiền vay** ·
19 Cơ cấu lại thời hạn trả nợ · **20 Nợ quá hạn** ·
**21 Chấm dứt cho vay, xử lý nợ, miễn, giảm lãi tiền vay, phí** · 22 Quy định nội bộ ·
23 Thỏa thuận cho vay · **24 Kiểm tra sử dụng tiền vay** · 25 Phạt vi phạm và bồi thường thiệt hại ·
26 Các quy định khác.
Chương II — Mục 1 (cho vay phục vụ hoạt động kinh doanh): 27 Phương thức cho vay ·
**28 Thời hạn cho vay** · 29 Lưu giữ hồ sơ cho vay *(bãi bỏ bởi TT 12/2024)*.
Mục 2 (cho vay phục vụ nhu cầu đời sống): 30 Phương thức cho vay · 31 Thời hạn cho vay ·
32 Lưu giữ hồ sơ cho vay *(bãi bỏ bởi TT 12/2024)*.
Mục 3 (cho vay bằng phương tiện điện tử, bổ sung bởi TT 06/2023): Điều 32a–32…
Chương III — 33 Hiệu lực thi hành · 34 Quy định chuyển tiếp · 35 Tổ chức thực hiện.

The consolidated text **does not renumber** — VBHN 21/2024 keeps the 2016 numbering and marks
amended clauses with superscript footnote markers (`3.10`, `2.18`, `5.11 (được bãi bỏ)`) plus
footnotes naming the amending circular and its effective date.

---

## 3. Passage → real article mapping (the deliverable)

Verbatim quotes are from `21_VBHN_NHNN_2024_hop_nhat_TT39_2016.pdf` unless marked
*(original)*. The 2016 original's text layer inserts spurious intra-word spaces
(`tổ ch ức tín d ụng`); whitespace must be normalised before any string matching.

### Q052 — claimed `Điều 13` → **CORRECT: Điều 13. Lãi suất cho vay** (khoản 1 + khoản 2)

> "1. Tổ chức tín dụng và khách hàng thỏa thuận về lãi suất cho vay theo cung cầu vốn thị trường,
> nhu cầu vay vốn và mức độ tín nhiệm của khách hàng, trừ trường hợp Ngân hàng Nhà nước Việt Nam
> có quy định về lãi suất cho vay tối đa tại khoản 2 Điều này."
>
> "2. Trường hợp khách hàng được tổ chức tín dụng đánh giá là có tình hình tài chính minh bạch,
> lành mạnh, tổ chức tín dụng và khách hàng thỏa thuận về lãi suất cho vay ngắn hạn bằng đồng
> Việt Nam nhưng không vượt quá mức lãi suất cho vay tối đa do Thống đốc Ngân hàng Nhà nước Việt
> Nam quyết định trong từng thời kỳ nhằm đáp ứng một số nhu cầu vốn: a) Phục vụ lĩnh vực phát
> triển nông nghiệp, nông thôn…; b) Thực hiện phương án kinh doanh hàng xuất khẩu…; c) Phục vụ
> kinh doanh của doanh nghiệp nhỏ và vừa…; d) Phát triển ngành công nghiệp hỗ trợ…; đ) Phục vụ
> kinh doanh của doanh nghiệp ứng dụng công nghệ cao…"

khoản 2 was **amended by TT 06/2023** (VBHN footnote 18), which added the gate
*"Trường hợp khách hàng được tổ chức tín dụng đánh giá là có tình hình tài chính minh bạch, lành mạnh"*.
The gold `reference_answer` ("Do TCTD và khách hàng thỏa thuận, không vượt mức trần lãi suất NHNN
quy định đối với vay ngắn hạn phục vụ nhu cầu vốn thiết yếu") **omits that gate**, and the phrase
"nhu cầu vốn thiết yếu" appears in neither version — the statute says "nhằm đáp ứng một số nhu cầu vốn"
and then enumerates five priority sectors. Article number is right; the answer text is a lossy
paraphrase that matches the *original* framing better than the consolidated one.

### Q053 — claimed `Điều 7` → **CORRECT: Điều 7. Điều kiện vay vốn** (khoản 1–4)

> "Tổ chức tín dụng xem xét, quyết định cho vay khi khách hàng có đủ các điều kiện sau đây:
> 1. Khách hàng là pháp nhân có năng lực pháp luật dân sự theo quy định của pháp luật. Khách hàng
> là cá nhân từ đủ 18 tuổi trở lên có năng lực hành vi dân sự đầy đủ theo quy định của pháp luật
> hoặc từ đủ 15 tuổi đến chưa đủ 18 tuổi không bị mất hoặc hạn chế năng lực hành vi dân sự theo
> quy định của pháp luật. 2. Nhu cầu vay vốn để sử dụng vào mục đích hợp pháp. 3. Có phương án sử
> dụng vốn khả thi. Điều kiện này không bắt buộc đối với khoản cho vay có mức giá trị nhỏ.
> 4. Có khả năng tài chính để trả nợ. 5. (được bãi bỏ)."

khoản 1, 2 and 4 are **unchanged** between original and consolidated. Two amendments touch this article:
**khoản 3 amended by TT 12/2024** (adds the small-value carve-out), and **khoản 5 bãi bỏ by TT 06/2023**
(`Điều 2. Bãi bỏ khoản 5 Điều 7.`). The gold answer lists exactly the four core conditions and mentions
neither change, so it is accurate against both versions but incomplete against the consolidated one.

### Q054 — claimed `Điều 21` → **PARTLY CORRECT. The rule spans Điều 24 + Điều 21 khoản 1.**

The `reference_answer` is "Kiểm tra giám sát sử dụng vốn đúng mục đích. Phát hiện sai mục đích thì có
quyền thu hồi nợ trước hạn." Those are two different provisions:

*inspection duty* → **Điều 24. Kiểm tra sử dụng tiền vay** (consolidated; amended twice):
> "1. Khách hàng có nghĩa vụ sử dụng vốn vay đúng mục đích đã cam kết, hoàn trả nợ gốc, lãi, phí đầy
> đủ, đúng hạn theo thỏa thuận; báo cáo việc sử dụng vốn vay và cung cấp thông tin, tài liệu, dữ liệu
> chứng minh vốn vay được sử dụng đúng mục đích.
> 2. Tổ chức tín dụng có quyền, nghĩa vụ kiểm tra, giám sát việc sử dụng vốn vay và trả nợ của khách
> hàng quy định tại khoản 1 Điều 102 Luật Các tổ chức tín dụng; có quyền yêu cầu khách hàng báo cáo
> việc sử dụng vốn vay và cung cấp thông tin, tài liệu, dữ liệu chứng minh vốn vay được sử dụng đúng
> mục đích.
> 3. Đối với các khoản cho vay có mức giá trị nhỏ, tổ chức tín dụng có biện pháp kiểm tra, giám sát
> việc sử dụng vốn vay đúng mục đích đã cam kết và trả nợ của khách hàng…"

For contrast, the **2016 original** Điều 24 read:
> *(original)* "1. Khách hàng có **trách nhiệm** sử dụng vốn vay và trả nợ theo nội dung thỏa thuận;
> báo cáo và cung cấp tài liệu chứng minh việc sử dụng vốn vay theo yêu cầu của tổ chức tín dụng.
> 2. Tổ chức tín dụng **có quyền thực hiện** kiểm tra, giám sát việc sử dụng vốn vay, trả nợ của khách
> hàng theo quy trình nội bộ quy định tại điểm c khoản 2 Điều 22 Thông tư này."

The original contains no "đúng mục đích" in Điều 24 and frames inspection as a *right*, not a
*right-and-duty*. Amendment chain: khoản 2 amended by TT 06/2023 → whole article replaced by TT 12/2024
(VBHN footnote 31, effective 01/7/2024).

*early recall* → **Điều 21 khoản 1** (**unamended**, byte-identical in both versions):
> "1. Tổ chức tín dụng có quyền chấm dứt cho vay, thu hồi nợ trước hạn theo nội dung đã thỏa thuận khi
> phát hiện khách hàng cung cấp thông tin sai sự thật, vi phạm quy định trong thỏa thuận cho vay
> và/hoặc hợp đồng bảo đảm tiền vay…"

Note Điều 21.1 triggers on "vi phạm quy định trong thỏa thuận cho vay" (misuse of capital breaches the
loan agreement) — it does not itself say "sai mục đích". So the claimed `Điều 21` is right for the second
half of the gold answer and wrong for the first. **This is the one passage whose wording favours the
consolidated text** ("đúng mục đích": 2 occurrences in the original, 8 in the VBHN).

### Q056 — claimed `Điều 18` → **WRONG. The rule is not in TT 39/2016 at all.**

`Điều 18` is **"Trả nợ gốc và lãi tiền vay"** (repayment of principal and interest) — nothing to do with
disbursement. Decisive evidence: a full-text search of both the 2016 original and VBHN 21/2024 returns
**0 occurrences of "tiền mặt" and 0 of "chuyển khoản"**.

TT 39/2016 only *delegates* the topic:
- *(original)* Điều 26.2: "Sử dụng các phương tiện thanh toán để giải ngân vốn cho vay theo quy định của
  Ngân hàng Nhà nước Việt Nam về việc sử dụng các phương tiện thanh toán để giải ngân vốn cho vay của tổ
  chức tín dụng đối với khách hàng." — reworded by TT 06/2023 to "…về **phương thức** giải ngân vốn cho vay…".
- Điều 23.1.h makes "Giải ngân vốn cho vay và việc sử dụng phương tiện thanh toán để giải ngân vốn cho vay"
  a mandatory term of the loan agreement.

The substantive rule lives in **Thông tư 21/2017/TT-NHNN** ngày 29/12/2017, "quy định về phương thức giải
ngân vốn cho vay của tổ chức tín dụng, chi nhánh ngân hàng nước ngoài đối với khách hàng", hiệu lực
02/04/2018 (it replaced TT 09/2012/TT-NHNN):
- **Điều 4. Phương thức giải ngân vốn cho vay sử dụng dịch vụ thanh toán không dùng tiền mặt** —
  "1. Tổ chức tín dụng cho vay phải sử dụng dịch vụ thanh toán không dùng tiền mặt theo quy định của
  pháp luật để giải ngân vốn cho vay vào tài khoản thanh toán của bên thụ hưởng tại tổ chức cung ứng
  dịch vụ thanh toán, trừ trường hợp quy định tại khoản 2 Điều này…"
- **Điều 5. Phương thức giải ngân vốn cho vay bằng tiền mặt** —
  "1. Tổ chức tín dụng cho vay được xem xét quyết định giải ngân vốn cho vay bằng tiền mặt trong các
  trường hợp: a) Khách hàng thanh toán, chi trả cho bên thụ hưởng (không bao gồm pháp nhân) không có tài
  khoản thanh toán tại tổ chức cung ứng dịch vụ thanh toán; b) Khách hàng là bên thụ hưởng (không bao gồm
  pháp nhân) không có tài khoản thanh toán…, đã ứng vốn tự có để thanh toán, chi trả các chi phí thuộc
  chính phương án, dự án kinh doanh hoặc phương án, dự án phục vụ đời sống…
  2. Khách hàng phải gửi cho tổ chức tín dụng cho vay văn bản cam kết của bên thụ hưởng về việc bên thụ
  hưởng không có tài khoản thanh toán…"
- **Điều 6** allows either method below a 100,000,000 VND threshold and for state-fund beneficiaries.

Two separate defects in this gold row: (a) the cited document does not contain the rule; (b) even against
TT 21/2017, the gold answer's test — "tiền mặt chỉ khi khách hàng có nhu cầu sử dụng tiền mặt hợp lý" —
**is not the statutory test**. The statute keys cash disbursement to the *beneficiary having no payment
account* (plus a written commitment), not to the borrower's "reasonable need for cash". This row should be
re-cited to TT 21/2017 Điều 4 + Điều 5 (+ Điều 6) and the reference answer rewritten, or dropped as unanswerable.

### Q057 — claimed `Điều 10` → **WRONG. Correct is Điều 28. Thời hạn cho vay** (unamended).

`Điều 10` is **"Loại cho vay"** — the short/medium/long-term classification
("1. Cho vay ngắn hạn là các khoản vay có thời hạn cho vay tối đa 01 (một) năm. 2. Cho vay trung hạn…
trên 01 (một) năm và tối đa 05 (năm) năm. 3. Cho vay dài hạn… trên 05 (năm) năm."). Topically adjacent,
but it does not state the basis for *setting* a term.

> **Điều 28. Thời hạn cho vay** — "1. Tổ chức tín dụng và khách hàng căn cứ vào chu kỳ hoạt động kinh
> doanh, thời hạn thu hồi vốn, khả năng trả nợ của khách hàng, nguồn vốn cho vay và thời hạn hoạt động
> còn lại của tổ chức tín dụng để thỏa thuận về thời hạn cho vay."

This matches the gold answer element-for-element (chu kỳ SXKD / thu hồi vốn / khả năng trả nợ / nguồn vốn
TCTD). Identical in original and consolidated; "chu kỳ hoạt động kinh doanh" occurs 4× in both.
Caveat: if the question is read as consumer lending, the parallel provision is **Điều 31**
(Mục 2), which uses a shorter basis ("khả năng trả nợ của khách hàng, nguồn vốn cho vay, thời hạn hoạt
động còn lại của tổ chức tín dụng"). The gold answer's mention of "dự án" points to Điều 28.

### Q058 — claimed `Điều 14` → **WRONG. Correct is Điều 20. Nợ quá hạn** (unamended), with the rate at Điều 13.4.c.

`Điều 14` is **"Phí liên quan đến hoạt động cho vay"** (early-repayment fee, standby-limit fee, syndication
arrangement fee, commitment fee, other fees) — unrelated.

> **Điều 20. Nợ quá hạn** — "Tổ chức tín dụng chuyển nợ quá hạn đối với số dư nợ gốc mà khách hàng không
> trả được nợ đúng hạn theo thỏa thuận và không được tổ chức tín dụng chấp thuận cơ cấu lại thời hạn trả
> nợ; thông báo cho khách hàng về việc chuyển nợ quá hạn. Nội dung thông báo tối thiểu bao gồm số dư nợ gốc
> bị quá hạn, thời điểm chuyển nợ quá hạn và lãi suất áp dụng đối với dư nợ gốc bị quá hạn."

Identical in original and consolidated ("chuyển nợ quá hạn đối với số dư nợ gốc": 3× in both).

The gold answer's second half — "áp dụng lãi suất nợ quá hạn" — is **Điều 13 khoản 4 điểm c** (unamended):
> "c) Trường hợp khoản nợ vay bị chuyển nợ quá hạn, thì khách hàng phải trả lãi trên dư nợ gốc bị quá hạn
> tương ứng với thời gian chậm trả, lãi suất áp dụng không vượt quá 150% lãi suất cho vay trong hạn tại
> thời điểm chuyển nợ quá hạn."

Two wording problems in the gold answer: it says "chuyển **toàn bộ** dư nợ gốc sang nợ quá hạn", whereas
Điều 20 converts only "số dư nợ gốc mà khách hàng không trả được nợ đúng hạn"; and it says "theo **hợp đồng
tín dụng**", whereas TT 39/2016 deliberately replaced that pre-2017 term with "**thỏa thuận cho vay**"
(it survives in TT 39 only in Điều 34, for contracts predating 15/3/2017). "Hợp đồng tín dụng" is
QĐ 1627/2001/QĐ-NHNN vocabulary — TT 39 repealed QĐ 1627 at Điều 33.2.a. Also relevant: Điều 18.3 is the
cross-reference that routes a borrower who cannot pay on time to either Điều 19 (restructuring) or
Điều 20 (overdue), and Điều 18.4 (amended by TT 06/2023) fixes the principal-before-interest recovery order.

### Summary table

| Q | Claimed | Real article(s) in TT 39/2016 | Claim # | Amended since 2016? |
|---|---|---|---|---|
| Q052 | Điều 13 | **Điều 13** khoản 1 + 2 | ✅ correct | khoản 2 amended by TT 06/2023 |
| Q053 | Điều 7 | **Điều 7** khoản 1–4 | ✅ correct | khoản 3 by TT 12/2024; khoản 5 repealed by TT 06/2023 |
| Q054 | Điều 21 | **Điều 24** (inspection) + **Điều 21.1** (early recall) | ⚠️ half-right | Điều 24 by TT 06/2023 then TT 12/2024; Điều 21 unamended |
| Q056 | Điều 18 | ❌ **not in this instrument** → TT 21/2017 Điều 4 + 5 (+6) | ❌ wrong | n/a |
| Q057 | Điều 10 | **Điều 28** (Điều 10 = "Loại cho vay") | ❌ wrong | Điều 28 unamended |
| Q058 | Điều 14 | **Điều 20** (+ Điều 13.4.c for the rate); Điều 14 = fees | ❌ wrong | Điều 20 unamended |

---

## 4. Amendment history of TT 39/2016 — authoritative list

Taken from the **preamble of the official VBHN 21/VBHN-NHNN** (Công báo số 875 + 876 ngày 29-7-2024),
which states the chain verbatim, cross-checked against each amending circular on congbao.

| Instrument | Date | Effective | What it does to TT 39/2016 |
|---|---|---|---|
| **QĐ 312/QĐ-NHNN** | 14/03/2017 | — | *Đính chính* (correction) only: Điều 8 khoản 5 (two wordings), Điều 8 khoản 6, Điều 29 điểm c khoản 1. **Touches none of the six articles.** |
| **TT 06/2023/TT-NHNN** | 28/06/2023 | 01/09/2023 | Điều 1: amends Điều 2 (điểm c khoản 6; adds khoản 12), Điều 8, Điều 11.2, **Điều 13.2**, Điều 18.4, Điều 22 (khoản 1; điểm a, b, c, e, g khoản 2), Điều 23.4.b, **Điều 24.2**, Điều 26 (khoản 2; adds khoản 5), Điều 27 (khoản 1, 4, 5); adds **Mục 3 Chương II** (Điều 32a–32…, cho vay bằng phương tiện điện tử). Điều 2: **repeals Điều 7 khoản 5**. |
| **TT 10/2023/TT-NHNN** | 23/08/2023 | 01/09/2023 | *Ngưng hiệu lực* (suspension) only: suspends **Điều 8 khoản 8, 9, 10** (the clauses TT 06/2023 had added) from 01/09/2023 until a new instrument governs them. Touches none of the six articles. |
| **TT 12/2024/TT-NHNN** | 28/06/2024 | 01/07/2024 | Điều 1: amends Điều 2 (khoản 1; adds khoản 13, 14), Điều 4.2, **Điều 7.3**, Điều 9, Điều 16.2, Điều 22 (điểm b(iii), c(iii) khoản 2 and adds a điểm c…), **Điều 24 (whole)**, Điều 26 (khoản 1, 3; adds khoản 5, 6, 7). Điều 2: **repeals Điều 29, Điều 32** and Điều 32g (as added by TT 06/2023), and repeals khoản 8 + điểm b khoản 9 Điều 1 of TT 06/2023. |
| **TT 52/2025/TT-NHNN** | 25/12/2025 | 25/12/2025 | Administrative only: Điều 1 amends **Điều 22 khoản 3** (which NHNN unit receives internal lending rules — now "NHNN chi nhánh Khu vực" / "Cục Quản lý, giám sát tổ chức tín dụng"); Điều 2 amends **Điều 35 khoản 2**. **Touches none of the six articles.** |

**The parent's four candidates are ruled out.** `14/2020`, `06/2020`, `22/2019` and `46/2020` do **not**
amend TT 39/2016 — the official VBHN preamble lists exactly three amending instruments
(06/2023, 10/2023, 12/2024), and TT 36/2018 (which appears in aggregator sidebars as a *related* document)
is likewise not in the chain. Consolidated texts that exist:
**VBHN 18/VBHN-NHNN (2023)** (post-06/2023), **VBHN 21/VBHN-NHNN (2024)** (post-12/2024, the one saved here),
and **VBHN 06/VBHN-NHNN (2026)** (adds TT 52/2025) — located on luatvietnam/thuvienphapluat; **its congbao
id was not resolved and it was not downloaded.**

---

## 5. Do the six passages quote *amended* wording? — original vs consolidated

**No passage quotes amended wording verbatim.** All six `gold_passage` / `reference_answer` strings are
loose Vietnamese paraphrases, not quotations. Measured discriminators (whole-document counts,
whitespace-normalised):

| Phrase | 2016 original | VBHN 21/2024 |
|---|---|---|
| `tình hình tài chính minh bạch, lành mạnh` (TT 06/2023 addition to Điều 13.2) | 1 | 2 |
| `mức giá trị nhỏ` (TT 12/2024 addition to Điều 7.3 / Điều 24.3) | 0 | 6 |
| `Điều 102 Luật Các tổ chức tín dụng` (TT 12/2024 Điều 24.2) | 0 | 2 |
| `có quyền, nghĩa vụ kiểm tra, giám sát` (amended Điều 24.2) | 0 | 1 |
| `có quyền thực hiện kiểm tra, giám sát` (**original** Điều 24.2) | 1 | 0 |
| `có trách nhiệm sử dụng vốn vay` (**original** Điều 24.1) | 1 | 0 |
| `có nghĩa vụ sử dụng vốn vay đúng mục đích` (amended Điều 24.1) | 0 | 1 |
| `đúng mục đích` | 2 | 8 |
| `chuyển nợ quá hạn đối với số dư nợ gốc` (Điều 20, Q058) | 3 | 3 |
| `chu kỳ hoạt động kinh doanh` (Điều 28, Q057) | 4 | 4 |
| `tiền mặt` / `chuyển khoản` (Q056) | 0 / 0 | 0 / 0 |

Reading:
- **Q057 and Q058 are version-independent** — Điều 28 and Điều 20 are byte-identical in both texts.
- **Q052 and Q053 lean ORIGINAL**: both gold answers omit the amendments' added qualifiers
  (the 06/2023 transparency gate in Điều 13.2, the 12/2024 small-value carve-out in Điều 7.3),
  and Q053 does not mention the repealed khoản 5. They are nonetheless *correct* against both versions,
  just incomplete against the consolidated one.
- **Q054 leans CONSOLIDATED**: "đúng mục đích" is the amended Điều 24's own wording; the original Điều 24
  never uses it and frames inspection as a right rather than a duty.
- **Q056 is unanswerable from either.**

### Recommendation

**Ingest `21_VBHN_NHNN_2024_hop_nhat_TT39_2016.pdf` (VBHN 21/VBHN-NHNN 2024) as the primary corpus
document; keep `39_2016_TT_NHNN.pdf` only as an explicitly version-tagged historical document, or drop it.**

Reasoning:
1. **Article numbers are stable** across both texts for every article the six rows cite (7, 10, 13, 14, 18,
   20, 21, 24, 28). VBHN does not renumber. So the gold `article_id` values that are correct stay correct
   under either choice — there is no renumbering hazard either way.
2. The consolidated text is the law in force. Ingesting only the 2016 original would mean a RAG model
   correctly retrieving **repealed** wording for Điều 24 ("có quyền thực hiện kiểm tra, giám sát…"),
   Điều 7 khoản 5 (bãi bỏ) and Điều 13 khoản 2 (missing the transparency gate) — and being scored as
   faithful while reciting dead law. That is the inverse failure mode, but it is still a wrong gold signal.
3. **Do not ingest both untagged.** They would emit near-duplicate chunks for Điều 7, 13, 18, 22, 24, 26, 27
   with contradictory text, so retrieval becomes a coin flip and both hallucination and citation-accuracy
   scores degrade for reasons unrelated to the models under test. If both are kept, carry a version field in
   the chunk metadata and pin the gold rows to one version.
4. VBHN 21/2024 is current through TT 12/2024 (01/7/2024). The only later amendment, TT 52/2025
   (25/12/2025), changes Điều 22.3 and Điều 35.2 — **neither is cited by any of the six rows** — so
   chasing VBHN 06/VBHN-NHNN (2026) is not required to fix these gold labels. Grab it only if the corpus
   is meant to be represented as "current as of 2026".

### Suggested gold-label fixes (for the parent to apply — no CSV/JSON was modified here)

- Q052: keep `article_id=13`. Optionally tighten the reference answer to include the 06/2023 gate and to
  replace "nhu cầu vốn thiết yếu" with the five enumerated priority sectors.
- Q053: keep `article_id=7`. Optionally note the TT 12/2024 small-value exception to khoản 3.
- Q054: change `article_id` to **24**, or split into two citations (24 for inspection, 21.1 for early recall).
  `21` alone does not support "kiểm tra giám sát đúng mục đích".
- Q056: **the cited document does not contain this rule.** Either re-cite to TT 21/2017/TT-NHNN
  Điều 4 + Điều 5 (+ Điều 6) — file already saved as `21_2017_TT_NHNN_phuong_thuc_giai_ngan.pdf` — and
  rewrite the answer around "beneficiary has no payment account", or reclassify the row as unanswerable.
  Leaving it as-is guarantees a false hallucination label.
- Q057: change `article_id` from `10` to **28** (or 31 if the consumer-lending reading is intended).
- Q058: change `article_id` from `14` to **20**; add **13.4.c** if the overdue-interest half of the answer
  is to be scored. Replace "hợp đồng tín dụng" with "thỏa thuận cho vay" and drop "toàn bộ".
- All six rows: `doc_link` / `gold_doc_ids` must move off `06/2023/TT-NHNN`. `doc_id_confidence` is
  currently `manifest`, i.e. these were never independently verified.

---

## 6. Gaps and unverified items

- **VBHN 06/VBHN-NHNN (2026)** exists (luatvietnam 425182, thuvienphapluat 692624) and consolidates
  TT 52/2025, but its congbao id and official PDF were **not** located; not downloaded. Not needed for these
  six rows (§5 point 4).
- **VBHN 18/VBHN-NHNN (2023)** — the post-06/2023 consolidated state — was **not** downloaded from an
  official source. Only the aggregator HTML was seen. Not needed.
- **TT 21/2017's current force status**: no repealing or amending instrument was found, and congbao's
  relationship graph confirms it *replaced* TT 09/2012/TT-NHNN. This is a negative search result, not a
  positive confirmation of "in force" — re-verify before it becomes a gold label.
- The congbao `so-do-van-ban-so-39-2016-tt-nhnn-22260` relationship graph is JS-rendered; a CLI fetch
  returns the plain document page instead. The amendment chain above was therefore taken from the VBHN
  preamble and each circular's own text, which is stronger evidence anyway.
- The congbao listing page `/van-ban-dang-cong-bao/ngan-hang-nha-nuoc-viet-nam-c7.htm` serves only the
  10 most recent NHNN documents as static HTML; older ids were found by search plus an id scan of
  `congbao.chinhphu.vn/van-ban/x-<id>.htm` (404 body = 2,803 bytes).
