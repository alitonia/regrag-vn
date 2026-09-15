"""Fill DOC_MANIFEST.json with URL -> instrument-id mappings verified from document text.

The 2026-09-10 CSV revision replaced self-describing `luatvietnam.vn` slugs with opaque
`datafiles.chinhphu.vn` filenames and `vanban.chinhphu.vn?docid=` links. canonical.py
resolves `gold.doc_id` from the manifest first and its slug regexes second, so 49 of 65
answerable rows collapsed to UNRESOLVED - not because the instruments are unknown, but
because nobody had written the new URLs into the manifest.

Those 49 rows collapse to 11 distinct URLs. This script resolves the ones whose identity
has been read off the delivered document itself (header text, born-digital or OCR'd) and
writes them in with the evidence attached. It is deliberately narrow:

  * a mapping is matched by a URL substring, then the EXACT url string is taken from the
    CSV - never retyped here, so a transcription error cannot poison gold.doc_id
  * a substring that matches nothing is a hard error, not a silent skip
  * an existing non-empty manifest value that disagrees is a hard error, not an overwrite
  * URLs whose document has not been read yet are left empty on purpose, so they keep
    surfacing as UNRESOLVED instead of being guessed

A wrong doc_id is worse than a missing one: it attaches the gold citation to an
instrument that does not cover the question, which manufactures false hallucination
labels across the whole benchmark.

Usage:
  python3 scripts/update_doc_manifest.py --dry-run
  python3 scripts/update_doc_manifest.py
"""

import argparse
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from regrag.corpus.canonical import split_urls

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MANIFEST_PATH = os.path.join(REPO_ROOT, "data", "raw_legal", "DOC_MANIFEST.json")
CSV_PATH = os.path.join(REPO_ROOT, "data", "gold", "bank_qa_data.csv")

# Each entry: the instrument id, a substring identifying the URL(s), and the evidence
# that was actually read off a delivered file. Nothing here is inferred from a filename
# alone.
VERIFIED = [
    {
        "doc_id": "32/2024/QH15",
        "url_contains": "docid=211190",
        "evidence": (
            "VERIFIED 2026-09-11 from the delivered file 32-2024-qh15_1.pdf, which has a "
            "clean born-digital text layer (viet_ratio 0.297, 192 'Điều' occurrences). Its "
            "first page reads: 'CÔNG BÁO/Số 369 + 370/Ngày 02-3-2024 / QUỐC HỘI / Luật số: "
            "32/2024/QH15 / LUẬT CÁC TỔ CHỨC TÍN DỤNG'. Both the bare docid=211190 URL and "
            "the &classid=1&typegroupid=3 variant point at this document. Supersedes the "
            "luatvietnam.vn slug, which was suspected of serving only page 1 of 209 articles; "
            "the delivered law is complete across two parts (94 + 59 pages)."
        ),
    },
    {
        "doc_id": "32/2013/TT-NHNN",
        "url_contains": "/2014/01/tt-32.pdf",
        "evidence": (
            "VERIFIED 2026-09-11 by re-OCR of the delivered tt-32.pdf (its embedded text layer "
            "is diacritic-stripped and unusable). tesseract 5.3.4 -l vie at 300dpi reads the "
            "header as: 'Số: 32/2013/TT-NHNN / Hà Nội, ngày 26 tháng 4 năm 2013 / THÔNG TƯ / "
            "Hướng dẫn thực hiện quy định hạn chế sử dụng ngoại hối trên lãnh thổ Việt Nam'. "
            "Corroborated independently by gold passage Q002, 'Điều 4. Các trường hợp được sử "
            "dụng ngoại hối trên lãnh thổ Việt Nam', which is this circular's subject matter. "
            "Path year 2014/01 is the publication month, not the instrument year."
        ),
    },
    {
        "doc_id": "06/2023/TT-NHNN",
        "url_contains": "/2023/7/06-nhnn.pdf",
        "evidence": (
            "VERIFIED 2026-09-11 by re-OCR of the delivered 06-2023-tt-nhnn.pdf (embedded text "
            "layer corrupt: viet_ratio 0.000, 0 'Điều' in 21,624 chars). tesseract reads: "
            "'Số: 06/2023/TT-NHNN / Hà Nội, ngày 28 tháng 6 năm 2023 / THÔNG TƯ / Sửa đổi, bổ "
            "sung một số điều của Thông tư số 39/2016/TT-NHNN'. Corroborated by gold passage "
            "Q049, 'Điều 1: Sửa đổi bổ sung một số điều của TT 39/2016 về hoạt động cho vay "
            "của TCTD'. IMPORTANT: this settles the 39/2016 amendment trap for Q049-Q058 - the "
            "questions cite the AMENDING circular, so 06/2023 is the text to ingest, not the "
            "2016 original and not a consolidated version."
        ),
    },
    {
        "doc_id": "46/2024/QH15",
        "url_contains": "/2025/01/luat46.pdf",
        "evidence": (
            "VERIFIED 2026-09-11 from the delivered luat46.pdf, clean born-digital text layer "
            "(viet_ratio 0.288, 147 'Điều'). First page reads: 'CÔNG BÁO/Số 1527 + 1528/Ngày "
            "29-12-2024 / QUỐC HỘI / Luật số: 46/2024/QH15 / LUẬT CÔNG CHỨNG'. Same instrument "
            "as the existing luatvietnam.vn slug entry; this URL is the complete text."
        ),
    },
    {
        "doc_id": "91/2015/QH13",
        "url_contains": "/2016/01/91.signed.pdf",
        "evidence": (
            "VERIFIED 2026-09-11 by re-OCR of the delivered 91.signed.pdf (94 pages, no text "
            "layer). The OCR text contains the headings 'BỘ LUẬT' and 'LUẬT DÂN SỰ' and, at the "
            "line corresponding to article 138, reads verbatim: 'Điều 138. Đại diện theo ủy "
            "quyền / 1. Cá nhân, pháp nhân có thể uỷ quyền cho cá nhân, pháp nhân khác xác...', "
            "which is exactly gold passage Q065 ('Điều 138 - Luật dân sự: Đại diện theo ủy "
            "quyền 1. Cá nhân, pháp nhân c...'). Bộ luật Dân sự 2015 is Luật số 91/2015/QH13, "
            "matching the filename's 91 and the 2016/01 publication path. This corrects an "
            "earlier suspicion that Q065 cited the wrong instrument: the citation is right, the "
            "manifest simply lacked the Civil Code."
        ),
    },
    {
        "doc_id": "06/2013/UBTVQH13",
        "url_contains": "/2013/08/6pl.pdf",
        "evidence": (
            "VERIFIED 2026-09-11 by re-OCR of the delivered 6pl.pdf. The header reads 'ỦY BAN "
            "THƯỜNG VỤ QUỐC HỘI ... Pháp lệnh số: 06/2013/UBTVQH13', i.e. the Pháp lệnh amending "
            "the Pháp lệnh Ngoại hối. Corroborated by the recitals of 32/2013/TT-NHNN, which cite "
            "'Pháp lệnh số 06/2013/PL-UBTVQH13 ngày 18 tháng 3 năm 2013 sửa đổi, bổ sung một số "
            "điều của Pháp lệnh Ngoại hối'."
        ),
    },
    {
        "doc_id": "41/2016/TT-NHNN",
        "url_contains": "16370-1-412016tt-nhnn16525pdf",
        "evidence": (
            "VERIFIED 2026-09-11 from data/raw_legal/41_2016_TT_NHNN.pdf (5,109,560 bytes, "
            "74 pages, born-digital, viet_ratio 0.279). Header reads 'Số: 41/2016/TT-NHNN "
            "... Hà Nội, ngày 30 tháng 12 năm 2016 ... Quy định tỷ lệ an toàn vốn đối với "
            "ngân hàng, chi nhánh ngân hàng nước ngoài'. Landing page congbao.chinhphu.vn id "
            "21991 (NOT 14181). Full provision map in data/raw_legal/TT41_FINDINGS.md. The "
            "Q029-Q038 gold rows were re-pointed to this URL on 2026-09-11: their old link "
            "(2024/7/22-nhnn.pdf) is TT 22/2024/TT-NHNN, a licensing circular with no "
            "capital-adequacy content."
        ),
    },
    {
        "doc_id": "22/2023/TT-NHNN",
        "url_contains": "/2024/01/22-nhnn.pdf",
        "evidence": (
            "VERIFIED 2026-09-11: downloaded this exact URL (23,767,020 bytes, sha1 "
            "50e2ac7f...), 50-page scan whose corrupt text layer reads 'so: LZ /2023ITT-NHNN "
            "... ngdy Z9 thdng,tZ ndm 2023 ... Sta d6i, b6 sung mQt s6 tli6u cia Th6ng tu s5 "
            "4U2016/TI-NHNN' = 'Số: 22/2023/TT-NHNN, ngày 29 tháng 12 năm 2023, Sửa đổi, bổ "
            "sung một số điều của Thông tư số 41/2016/TT-NHNN'; OCR at "
            "data/raw_legal/ocr/22_2023_TT_NHNN.ocr.txt. NOTE: the sibling URL "
            "2023/12/thong-tu-22-2023-tt-nhnn.pdf is a dead mirror, and the delivered file "
            "named 22-2023-tt-nhnn.pdf is TT 22/2024 (amends TT 50/2018) - see "
            "TT41_FINDINGS.md. Amends khoản 10/điểm b khoản 11 Điều 9 of TT 41/2016, i.e. "
            "the risk-weight grids Q032/Q034/Q035 depend on; must be ingested together "
            "with 41/2016, never alone."
        ),
    },
    {
        "doc_id": "39/2016/TT-NHNN",
        "url_contains": "87621-vbhn-nhnn.pdf",
        "evidence": (
            "VERIFIED 2026-09-11 from data/raw_legal/21_VBHN_NHNN_2024_hop_nhat_TT39_2016.pdf "
            "(829,233 bytes, 36 pages, born-digital, viet_ratio 0.284, 123 'Điều'). Header: "
            "'VBHN 21/VBHN-NHNN' consolidated text of TT 39/2016/TT-NHNN current through TT "
            "12/2024 (Công báo 875+876, 29-7-2024). Article numbering is NOT renumbered vs "
            "the 2016 original, so gold article ids stay valid. This URL is the consolidated "
            "attachment (congbao id 42346). Ingest the VBHN as the single 39/2016 corpus "
            "document; do NOT also ingest the 2016 original untagged (near-duplicate "
            "contradictory chunks). Amendment chain per the VBHN preamble: QĐ 312/2017 "
            "(đính chính), TT 06/2023, TT 10/2023 (suspends Điều 8.8-10), TT 12/2024, TT "
            "52/2025 (Điều 22.3, 35.2 only). See TT39_FINDINGS.md."
        ),
    },
    {
        "doc_id": "21/2017/TT-NHNN",
        "url_contains": "38621-2017-tt-nhnn.pdf",
        "evidence": (
            "VERIFIED 2026-09-11 from data/raw_legal/21_2017_TT_NHNN_phuong_thuc_giai_ngan.pdf "
            "(306,437 bytes, 7 pages, born-digital, viet_ratio 0.258). Header: 'Số: "
            "21/2017/TT-NHNN ... quy định về phương thức giải ngân vốn cho vay của tổ chức "
            "tín dụng, chi nhánh ngân hàng nước ngoài đối với khách hàng'. Needed for Q056: "
            "the cash/cashless disbursement rule (Điều 4, 5, 6) is NOT in TT 39/2016 at all "
            "('tiền mặt'/'chuyển khoản' occur 0 times in both 39/2016 texts). See "
            "TT39_FINDINGS.md §3."
        ),
    },
    {
        "doc_id": "22/2018/TT-NHNN",
        "url_contains": "/2019/01/49-nhnn.pdf",
        "evidence": (
            "VERIFIED 2026-09-11 by re-OCR of the delivered 49-nhnn.pdf (8 pages, no text "
            "layer). tesseract reads the header as 'NGÂN HÀNG NHÀ NƯỚC VIỆT NAM ... Số: "
            "22/2018/TT-NHNN Hà Nội, ngày ... năm 2018' - the term-deposit circular "
            "('tiền gửi có kỳ hạn'), NOT number 49/2018 and NOT 49/2019: Q065's answer "
            "cited both of those non-existent numbers and was corrected 2026-09-11 to "
            "22/2018/TT-NHNN. Path 2019/01 is the publication month. Third distinct '22' "
            "circular in the delivery (22/2018, 22/2023, 22/2024); ingested as a "
            "retrieval distractor."
        ),
    },
    {
        "doc_id": "47/2014/QH13",
        "url_contains": "62-vbhn-vpqh.pdf",
        "evidence": (
            "VERIFIED 2026-09-11 by re-OCR of the delivered 62-vbhn-vpqh.pdf (36 pages, no "
            "text layer, ocr_viet_ratio 0.297, 176 'Điều'). The OCR reads: 'LUẬT NHẬP CẢNH, "
            "XUẤT CẢNH, QUÁ CẢNH, CƯ TRÚ CỦA NGƯỜI NƯỚC NGOÀI TẠI VIỆT NAM ... Luật ... số "
            "47/2014/QH13 ngày 16 tháng 6 năm 2014 ..., được sửa đổi, bổ sung bởi: 1. Luật "
            "số 51/2019/QH14' - i.e. the CONSOLIDATED text (văn bản hợp nhất) of the "
            "immigration law including the 51/2019/QH14 amendments; cite as consolidated. "
            "Resolves Q008, whose passage quotes Điều 3 khoản 11-13 definitions."
        ),
    },
    {
        "doc_id": "06/2019/TT-NHNN",
        "url_contains": "27480-1-2019611-61206-2019-tt-nhnn.pdf",
        "evidence": (
            "VERIFIED 2026-09-11. The delivered file '2019_611 + 612_06-2019-TT-NHNN.pdf' is "
            "byte-identical (sha1) to '06-2019-tt-nhnn.pdf', whose text layer is clean "
            "(viet_ratio 0.301, 37 'Điều') and reads 'CÔNG BÁO/Số 611 + 612/Ngày 03-8-2019 ... "
            "Số: 06/2019/TT-NHNN / Hà Nội, ngày 26 tháng 6 năm 2019 / THÔNG TƯ / Hướng dẫn về "
            "quản lý ngoại hối đối với hoạt động đầu tư trực tiếp nước ngoài vào Việt Nam'. "
            "The URL path /2019/6/29358/ carries the same van-ban id 29358 as the existing "
            "congbao.chinhphu.vn slug for this circular, and 2019611-61206 encodes CÔNG BÁO "
            "611+612. Three independent agreements."
        ),
    },
]


def csv_urls(csv_path: str):
    """Every URL in the CSV, with the question ids that cite it."""
    cites = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for i, row in enumerate(csv.DictReader(f), 1):
            if not (row.get("question") or "").strip():
                continue
            raw_id = (row.get("ID") or "").strip()
            qid = f"Q{int(raw_id):03d}" if raw_id.isdigit() else (raw_id or f"Q{i:03d}")
            for u in split_urls(row.get("doc_link") or ""):
                cites.setdefault(u, []).append(qid)
    return cites


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default=MANIFEST_PATH)
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--dry-run", action="store_true", help="Report what would change")
    args = ap.parse_args()

    if not os.path.exists(args.manifest):
        raise SystemExit(f"manifest not found: {args.manifest}")
    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    urls = manifest.setdefault("urls", {})
    evidence = manifest.setdefault("_evidence", {})

    cites = csv_urls(args.csv)
    errors, added, filled = [], [], []

    for entry in VERIFIED:
        needle = entry["url_contains"]
        hits = sorted(u for u in cites if needle in u)
        if not hits:
            errors.append(f"{entry['doc_id']}: no CSV url contains {needle!r}")
            continue
        for u in hits:
            existing = urls.get(u)
            if existing and existing != entry["doc_id"]:
                errors.append(
                    f"CONFLICT {u}\n    manifest already says {existing!r}, "
                    f"verification says {entry['doc_id']!r}"
                )
                continue
            n_q = len(cites[u])
            if existing == entry["doc_id"]:
                continue
            if u in urls:
                filled.append((u, entry["doc_id"], n_q))
            else:
                added.append((u, entry["doc_id"], n_q))
            if not args.dry_run:
                urls[u] = entry["doc_id"]
                evidence[entry["doc_id"]] = entry["evidence"]

    print("=== manifest resolution from verified document text ===\n")
    print(f"would ADD   {len(added)} new url mapping(s):")
    for u, d, n in added:
        print(f"  {d:<18} {n:>2} q  {u}")
    print(f"\nwould FILL  {len(filled)} existing-but-empty mapping(s):")
    for u, d, n in filled:
        print(f"  {d:<18} {n:>2} q  {u}")

    total_q = sum(n for _, _, n in added + filled)
    print(f"\nquestions gaining a resolved doc_id: {total_q}")

    still_empty = sorted(u for u, v in urls.items() if not v)
    print(f"\nstill UNRESOLVED on purpose ({len(still_empty)} url(s) - document not yet read):")
    for u in still_empty:
        print(f"  {len(cites.get(u, [])):>2} q  {u}")

    if errors:
        print("\nERRORS - nothing written for these:")
        for e in errors:
            print(f"  {e}")

    if args.dry_run:
        print("\n[dry-run] manifest not modified")
        return 1 if errors else 0

    if errors and not (added or filled):
        raise SystemExit("refusing to write: every mapping errored")

    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"\nwrote {os.path.relpath(args.manifest, REPO_ROOT)}")
    print("Now re-run: .venv/bin/python scripts/regenerate.py --force")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
