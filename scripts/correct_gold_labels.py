#!/usr/bin/env python3
"""Apply verified gold-label corrections to data/gold/bank_qa_data.csv.

Three defect classes were verified against primary sources on 2026-09-11
(see data/raw_legal/TT41_FINDINGS.md and TT39_FINDINGS.md for the verbatim
evidence; every rewritten cell below quotes or closely tracks text read off
the official PDFs saved in data/raw_legal/):

* Q029-Q038 (CAR block) all cite 22/2024/TT-NHNN (a licensing circular
  mislabelled "22-2023") while asking about TT 41/2016/TT-NHNN capital
  adequacy; all ten article locators are wrong and three gold answers
  (Q032, Q034, Q035) contradict the law. Q037's authority is TT 35/2015
  (biểu 118/119-TTGS), which is not in the corpus, so the row is converted
  to an explicit coverage-gap probe instead of being scored against a
  citation target no instrument in the corpus can support.
* Q050, Q051, Q055, Q057, Q058 cite amender clauses ("Điều 1 Khoản N" of
  TT 06/2023) or the wrong parent article for provisions that live in
  TT 39/2016 itself; Q056's rule is not in TT 39/2016 at all but in
  TT 21/2017/TT-NHNN Điều 4-6. Q049's answer is refined to the verified
  amendment list. Corrected lending rows are re-pointed to the official
  consolidated text (VBHN 21/VBHN-NHNN 2024), which keeps the 2016 article
  numbering, per the TT39 findings' ingest recommendation.
* Probe rows carry a blank ID cell, so their ids derive from row order;
  inserting any row above them silently renumbers (and thus invalidates)
  every cached probe generation. This script pins explicit ids 66-88.

Every correction is guarded: the script refuses to touch a row whose
current cells do not match the recorded "before" state, so a moved or
already-edited CSV fails loudly instead of being corrupted. The author
cell records the correction provenance and is carried into
GoldQuestion.author by the loader.
"""

import argparse
import csv
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(REPO_ROOT, "data", "gold", "bank_qa_data.csv")

U_TT41 = "https://congbaocdn.chinhphu.vn/CongBaoCP/VanBan/2016/12/21991/16370-1-412016tt-nhnn16525pdf"
U_TT22_2023 = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/01/22-nhnn.pdf"
U_VBHN_39 = "https://congbaocdn.chinhphu.vn/CongBaoCP/VanBan/2024/7/42346/51014-1-2024875-87621-vbhn-nhnn.pdf"
U_TT21_2017 = "https://congbaocdn.chinhphu.vn/CongBaoCP/VanBan/2017/12/25880/21382-1-2018385-38621-2017-tt-nhnn.pdf"
U_WRONG_22_2024 = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/7/22-nhnn.pdf"
U_TT06_2023 = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2023/7/06-nhnn.pdf"

NOTE_TT41 = "Thành; corrected 2026-09-11 per data/raw_legal/TT41_FINDINGS.md"
NOTE_TT39 = "Thành; corrected 2026-09-11 per data/raw_legal/TT39_FINDINGS.md"
#: Corrections discovered by the repair-pass guards (2026-09-11), verified
#: directly against the corpus chunks named in the passage.
NOTE_CORPUS_17 = (
    "Thành; corrected 2026-09-11 against TT 17/2024/TT-NHNN Điều 11 "
    "(the cited Điều 5 does not exist in this circular)"
)
NOTE_CORPUS_06 = (
    "Thành; corrected 2026-09-11 against TT 06/2019/TT-NHNN "
    "(Điều 15 is the organisation clause; the duties are Điều 11, the "
    "currency sale is Điều 6 khoản 2 điểm b)"
)
NOTE_Q037 = (
    "Thành; corrected 2026-09-11 per data/raw_legal/TT41_FINDINGS.md: authority is "
    "TT 35/2015/TT-NHNN (biểu 118/119-TTGS), not in the corpus - converted to "
    "coverage-gap probe"
)

# id -> (expected_before, after). Fields not listed in a dict are left unchanged.
CORRECTIONS = {
    "20": (
        {
            "doc_link": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/7/17-nhnn.pdf",
            "passage_prefix": "Điều 5: Cá nhân từ đủ 18 tuổi",
        },
        {
            "text_contains_answer_in_the_doc": (
                "Điều 11: Cá nhân mở tài khoản thanh toán bao gồm: người từ đủ 15 tuổi "
                "trở lên không bị hạn chế hoặc mất năng lực hành vi dân sự (điểm a khoản 1); "
                "người chưa đủ 15 tuổi, người bị hạn chế hoặc mất năng lực hành vi dân sự mở "
                "tài khoản thanh toán thông qua người đại diện theo pháp luật (điểm b); người "
                "có khó khăn trong nhận thức, làm chủ hành vi mở thông qua người giám hộ (điểm c)."
            ),
            "answer": (
                "Cá nhân từ đủ 15 tuổi trở lên, không bị hạn chế hoặc mất năng lực hành vi "
                "dân sự, được tự mở tài khoản thanh toán (điểm a khoản 1 Điều 11 TT "
                "17/2024/TT-NHNN). Người chưa đủ 15 tuổi, người bị hạn chế hoặc mất năng lực "
                "hành vi dân sự mở qua người đại diện theo pháp luật (điểm b); người có khó "
                "khăn trong nhận thức, làm chủ hành vi mở qua người giám hộ (điểm c)."
            ),
            "author": NOTE_CORPUS_17,
        },
    ),
    "29": (
        {
            "doc_link": U_WRONG_22_2024,
            "passage_prefix": "Điều 9: Ngân hàng phải duy trì tỷ lệ an toàn vốn tối thiểu 8%.",
        },
        {
            "doc_link": U_TT41,
            "text_contains_answer_in_the_doc": (
                "Điều 6: Ngân hàng không có công ty con, chi nhánh ngân hàng nước ngoài "
                "phải thường xuyên duy trì tỷ lệ an toàn vốn xác định trên cơ sở báo cáo "
                "tài chính của ngân hàng tối thiểu 8% (khoản 2); ngân hàng có công ty con "
                "phải duy trì tỷ lệ an toàn vốn riêng lẻ và hợp nhất tối thiểu 8% (khoản 3)."
            ),
            "author": NOTE_TT41,
        },
    ),
    "30": (
        {"doc_link": U_WRONG_22_2024, "passage_prefix": "Điều 5: Vốn tự có gồm vốn cấp 1"},
        {
            "doc_link": U_TT41,
            "text_contains_answer_in_the_doc": (
                "Điều 7: Vốn tự có bao gồm tổng Vốn cấp 1 và Vốn cấp 2 trừ đi các khoản "
                "giảm trừ quy định tại Phụ lục 1 ban hành kèm theo Thông tư này (khoản 2)."
            ),
            "answer": (
                "Vốn tự có = tổng Vốn cấp 1 và Vốn cấp 2 trừ đi các khoản giảm trừ quy "
                "định tại Phụ lục 1. Vốn cấp 1 gồm vốn điều lệ, quỹ dự trữ bổ sung vốn "
                "điều lệ, quỹ đầu tư phát triển nghiệp vụ, quỹ dự phòng tài chính, vốn "
                "đầu tư xây dựng cơ bản, lợi nhuận chưa phân phối, thặng dư vốn cổ phần, "
                "trừ lợi thế thương mại, lỗ lũy kế, cổ phiếu quỹ. Vốn cấp 2 gồm trái "
                "phiếu chuyển đổi, nợ thứ cấp, dự phòng chung."
            ),
            "author": NOTE_TT41,
        },
    ),
    "31": (
        {"doc_link": U_WRONG_22_2024, "passage_prefix": "Phụ lục: Hệ số rủi ro 0% cho tiền mặt"},
        {
            "doc_link": U_TT41,
            "text_contains_answer_in_the_doc": (
                "Điều 9: Hệ số rủi ro tín dụng là 0% đối với tài sản là tiền mặt, vàng "
                "và các khoản tương đương tiền mặt (khoản 2); khoản phải đòi Chính phủ "
                "Việt Nam, Ngân hàng Nhà nước, Kho bạc Nhà nước, Ủy ban nhân dân tỉnh, "
                "thành phố trực thuộc Trung ương, các ngân hàng chính sách (khoản 3)."
            ),
            "author": NOTE_TT41,
        },
    ),
    "32": (
        {"doc_link": U_WRONG_22_2024, "passage_prefix": "Phụ lục: Hệ số rủi ro 50% cho vay"},
        {
            "doc_link": U_TT41 + "\n" + U_TT22_2023,
            "text_contains_answer_in_the_doc": (
                "Điều 9: Hệ số rủi ro áp dụng cho khoản cho vay thế chấp nhà ở theo Tỷ "
                "lệ bảo đảm (LTV) và Tỷ lệ thu nhập (DSC) (điểm b khoản 11, sửa đổi, bổ "
                "sung bởi TT 22/2023/TT-NHNN)."
            ),
            "answer": (
                "Không có mức cố định theo một ngưỡng LTV duy nhất: hệ số rủi ro cho "
                "khoản cho vay thế chấp nhà ở xác định theo lưới hai chiều LTV × DSC "
                "tại điểm b khoản 11 Điều 9 TT 41/2016/TT-NHNN (sửa đổi bởi TT "
                "22/2023/TT-NHNN), dao động 20%-100%: LTV dưới 40% và DSC ≤ 35% là "
                "25%; LTV 60%-dưới 80% và DSC > 35% là 50%; LTV ≥ 100% và DSC > 35% là "
                "100%. Khoản cho vay mua nhà ở xã hội, nhà ở theo chương trình, dự án "
                "hỗ trợ của Chính phủ có lưới riêng 20%-50%."
            ),
            "author": NOTE_TT41,
        },
    ),
    "33": (
        {"doc_link": U_WRONG_22_2024, "passage_prefix": "Điều 8: RWA = rủi ro tín dụng"},
        {
            "doc_link": U_TT41,
            "text_contains_answer_in_the_doc": (
                "Điều 6: Tỷ lệ an toàn vốn CAR = C / (RWA + 12,5 × (KOR + KMR)) × 100%, "
                "trong đó C là vốn tự có, RWA là tổng tài sản tính theo rủi ro tín dụng, "
                "KOR là vốn yêu cầu cho rủi ro hoạt động, KMR là vốn yêu cầu cho rủi ro "
                "thị trường (khoản 1)."
            ),
            "answer": (
                "Ba loại rủi ro nhập vào tỷ lệ an toàn vốn theo công thức tại khoản 1 "
                "Điều 6: CAR = C / (RWA + 12,5 × (KOR + KMR)) × 100%. Riêng ký hiệu RWA "
                "chỉ gồm rủi ro tín dụng (Điều 8: RWA = RWACR + RWACCR); rủi ro hoạt "
                "động (KOR, Điều 16) và rủi ro thị trường (KMR, Điều 18) quy đổi qua hệ "
                "số 12,5 ở mẫu số."
            ),
            "author": NOTE_TT41,
        },
    ),
    "34": (
        {"doc_link": U_WRONG_22_2024, "passage_prefix": "Phụ lục: Hệ số rủi ro 100% cho vay DN"},
        {
            "doc_link": U_TT41 + "\n" + U_TT22_2023,
            "text_contains_answer_in_the_doc": (
                "Điều 9: Đối với các tài sản khác trên bảng cân đối kế toán, trừ các tài "
                "sản quy định tại khoản 1 đến khoản 17 Điều này, hệ số rủi ro tín dụng "
                "là 100% (khoản 18)."
            ),
            "answer": (
                "Mức 100% là hệ số rủi ro residual tại khoản 18 Điều 9, áp dụng cho các "
                "tài sản khác trên bảng cân đối kế toán không thuộc khoản 1-17 (gồm cả "
                "tài sản cố định, bất động sản đầu tư, các khoản phải thu khác). Riêng "
                "cho vay doanh nghiệp không dùng mức phẳng 100%: áp lưới doanh thu × đòn "
                "bẩy × vốn chủ sở hữu tại điểm b khoản 9 Điều 9 (50%-250%); doanh nghiệp "
                "nhỏ và vừa 90% (điểm a); doanh nghiệp không cung cấp báo cáo tài chính "
                "200%; doanh nghiệp hoạt động dưới 1 năm 150%."
            ),
            "author": NOTE_TT41,
        },
    ),
    "35": (
        {"doc_link": U_WRONG_22_2024, "passage_prefix": "Phụ lục: Hệ số rủi ro 150% cho vay"},
        {
            "doc_link": U_TT41 + "\n" + U_TT22_2023,
            "text_contains_answer_in_the_doc": (
                "Điều 9: Đối với khoản nợ xấu có dự phòng cụ thể nhỏ hơn 20% giá trị của "
                "khoản nợ xấu, hệ số rủi ro tín dụng là 150% (khoản 13 điểm a)."
            ),
            "answer": (
                "150% áp dụng cho: khoản nợ xấu có dự phòng cụ thể nhỏ hơn 20% giá trị "
                "khoản nợ (điểm a khoản 13 Điều 9); khoản phải đòi tổ chức quốc tế, khu "
                "vực tài chính quốc tế không được xếp hạng hoặc xếp hạng dưới B (khoản "
                "5); doanh nghiệp hoạt động dưới 1 năm (điểm b khoản 9); tài sản là góp "
                "vốn, cho vay mua chứng khoán kinh doanh, đầu tư chứng khoán, cho vay "
                "kinh doanh chứng khoán (khoản 15); khoản cấp tín dụng bảo đảm bằng bất "
                "động sản khi không có thông tin LTV (điểm đ khoản 10). Cho vay kinh "
                "doanh bất động sản theo lưới LTV: 75% (dưới 60%), 100% (60%-dưới 75%), "
                "120% (từ 75%) (điểm c khoản 10); cấp tín dụng tài trợ dự án kinh doanh "
                "bất động sản 200%, dự án khu công nghiệp 160% (điểm e khoản 10, sửa "
                "đổi bởi TT 22/2023). Thông tư dùng thuật ngữ \"nợ xấu\", không dùng "
                "\"nợ quá hạn trên 90 ngày\"."
            ),
            "author": NOTE_TT41,
        },
    ),
    "36": (
        {"doc_link": U_WRONG_22_2024, "passage_prefix": "Điều 6: Vốn cấp 2 tối đa bằng 100%"},
        {
            "doc_link": U_TT41,
            "text_contains_answer_in_the_doc": (
                "Phụ lục 1, Phần A, mục I: Giá trị vốn cấp 2 tối đa bằng vốn cấp 1 "
                "(VỐN CẤP 2 (B) = B1 - B2 - 20)."
            ),
            "author": NOTE_TT41,
        },
    ),
    "37": (
        {"doc_link": U_WRONG_22_2024, "passage_prefix": "Điều 14: Báo cáo tỷ lệ an toàn vốn"},
        {
            "doc_link": "",
            "text_contains_answer_in_the_doc": "",
            "answer": "Không có trong kho văn bản",
            "author": NOTE_Q037,
        },
    ),
    "38": (
        {"doc_link": U_WRONG_22_2024, "passage_prefix": "Điều 11: Vốn yêu cầu cho rủi ro"},
        {
            "doc_link": U_TT41,
            "text_contains_answer_in_the_doc": (
                "Điều 16: Vốn yêu cầu cho rủi ro hoạt động KOR = (BI năm thứ n + BI năm "
                "thứ n-1 + BI năm thứ n-2) / 3 × 15%, với BI là chỉ số kinh doanh xác "
                "định theo quý gần nhất (khoản 1, khoản 2)."
            ),
            "answer": (
                "Theo khoản 1 Điều 16 TT 41/2016/TT-NHNN: KOR = (BI năm thứ n + BI năm "
                "thứ n-1 + BI năm thứ n-2) / 3 × 15%. BI là chỉ số kinh doanh (Business "
                "Indicator) = IC + SC + FC (khoản 2, chi tiết tại Phụ lục 3), xác định "
                "theo quý gần nhất tại thời điểm tính toán - đây là công thức chỉ số "
                "kinh doanh của Basel III, không phải phương pháp chỉ số cơ bản (BIA) "
                "của Basel II."
            ),
            "author": NOTE_TT41,
        },
    ),
    "46": (
        {
            "doc_link": "https://congbaocdn.chinhphu.vn/CongBaoCP/VanBan/2019/6/29358/27480-1-2019611-61206-2019-tt-nhnn.pdf",
            "passage_prefix": "Điều 15: NH kiểm tra giám sát",
        },
        {
            "text_contains_answer_in_the_doc": (
                "Điều 11: Trách nhiệm của tổ chức tín dụng được phép: hướng dẫn doanh nghiệp "
                "có vốn đầu tư trực tiếp nước ngoài, nhà đầu tư nước ngoài xuất trình tài "
                "liệu, chứng từ hợp lệ (khoản 1); mở, đóng tài khoản vốn đầu tư trực tiếp "
                "theo đề nghị (khoản 2); xem xét, kiểm tra, lưu giữ giấy tờ, chứng từ phù hợp "
                "giao dịch thực tế để đảm bảo dịch vụ ngoại hối thực hiện đúng mục đích "
                "(khoản 3); yêu cầu cung cấp tài liệu, chứng từ liên quan (khoản 4); bán ngoại "
                "tệ để chuyển ra nước ngoài trên cơ sở tự cân đối nguồn ngoại tệ (khoản 5); "
                "xác nhận bằng văn bản số dư, thông tin giao dịch theo yêu cầu (khoản 6)."
            ),
            "answer": (
                "TCTD được phép có trách nhiệm: hướng dẫn xuất trình tài liệu, chứng từ hợp "
                "lệ về quản lý ngoại hối; mở, đóng tài khoản vốn đầu tư trực tiếp theo đề "
                "nghị; xem xét, kiểm tra, lưu giữ giấy tờ, chứng từ phù hợp giao dịch thực tế "
                "để đảm bảo cung ứng dịch vụ ngoại hối đúng mục đích; yêu cầu cung cấp tài "
                "liệu, chứng từ liên quan; bán ngoại tệ cho nhà đầu tư để chuyển ra nước "
                "ngoài trên cơ sở tự cân đối nguồn ngoại tệ; xác nhận bằng văn bản số dư, "
                "thông tin giao dịch theo yêu cầu (Điều 11 TT 06/2019/TT-NHNN)."
            ),
            "author": NOTE_CORPUS_06,
        },
    ),
    "47": (
        {
            "doc_link": "https://congbaocdn.chinhphu.vn/CongBaoCP/VanBan/2019/6/29358/27480-1-2019611-61206-2019-tt-nhnn.pdf",
            "passage_prefix": "Điều 9: Được bán ngoại tệ",
        },
        {
            "text_contains_answer_in_the_doc": (
                "Điều 6: Các giao dịch chi trên tài khoản vốn đầu tư trực tiếp: chi bán ngoại "
                "tệ cho tổ chức tín dụng được phép để chuyển vào tài khoản thanh toán bằng "
                "đồng Việt Nam của chính doanh nghiệp có vốn đầu tư trực tiếp nước ngoài, nhà "
                "đầu tư nước ngoài (khoản 2 điểm b)."
            ),
            "answer": (
                "Được. Doanh nghiệp có vốn đầu tư trực tiếp nước ngoài được chi bán ngoại tệ "
                "trên tài khoản vốn đầu tư trực tiếp cho tổ chức tín dụng được phép để chuyển "
                "vào tài khoản thanh toán bằng đồng Việt Nam của chính doanh nghiệp (điểm b "
                "khoản 2 Điều 6 TT 06/2019/TT-NHNN)."
            ),
            "author": NOTE_CORPUS_06,
        },
    ),
    "49": (
        {"doc_link": U_TT06_2023, "passage_prefix": "Điều 1: Sửa đổi bổ sung một số điều"},
        {
            "answer": (
                "Sửa đổi, bổ sung các điều 2, 8, 11, 13, 18, 22, 23, 24, 26, 27 của TT "
                "39/2016/TT-NHNN; bổ sung Mục 3 Chương II về cho vay bằng phương tiện "
                "điện tử (Điều 32a-32h); bãi bỏ khoản 5 Điều 7 (Điều 1, Điều 2 TT "
                "06/2023/TT-NHNN)."
            ),
            "author": NOTE_TT39,
        },
    ),
    "50": (
        {"doc_link": U_TT06_2023, "passage_prefix": "Điều 1 Khoản 2: TCTD xem xét năng lực"},
        {
            "doc_link": U_VBHN_39,
            "text_contains_answer_in_the_doc": (
                "Điều 7: Tổ chức tín dụng xem xét, quyết định cho vay khi khách hàng có "
                "đủ điều kiện: năng lực pháp luật dân sự, năng lực hành vi dân sự "
                "(khoản 1); nhu cầu vay vốn sử dụng vào mục đích hợp pháp (khoản 2); "
                "phương án sử dụng vốn khả thi, không bắt buộc đối với khoản cho vay có "
                "mức giá trị nhỏ (khoản 3); khả năng tài chính để trả nợ (khoản 4); "
                "khoản 5 đã được bãi bỏ."
            ),
            "author": NOTE_TT39,
        },
    ),
    "51": (
        {"doc_link": U_TT06_2023, "passage_prefix": "Điều 1 Khoản 3: Không cho vay để gửi"},
        {
            "doc_link": U_VBHN_39,
            "text_contains_answer_in_the_doc": (
                "Điều 8: Tổ chức tín dụng không được cho vay đối với các nhu cầu vốn: "
                "hoạt động đầu tư kinh doanh thuộc ngành, nghề cấm (khoản 1-3); mua vàng "
                "miếng (khoản 4); trả nợ khoản cấp tín dụng tại chính tổ chức tín dụng "
                "cho vay (khoản 5); trả nợ vay nước ngoài, khoản cấp tín dụng tại tổ "
                "chức tín dụng khác (khoản 6); gửi tiền (khoản 7); góp vốn, mua cổ phần "
                "chưa niêm yết (khoản 8); dự án không đủ điều kiện đưa vào kinh doanh "
                "(khoản 9; khoản 8, 9 hiện ngưng hiệu lực theo TT 10/2023/TT-NHNN)."
            ),
            "answer": (
                "Không được cho vay để: thực hiện hoạt động đầu tư kinh doanh thuộc "
                "ngành, nghề cấm theo Luật Đầu tư; mua vàng miếng; trả nợ khoản cấp tín "
                "dụng tại chính TCTD cho vay (trừ lãi tiền vay trong quá trình thi công "
                "được tính trong tổng mức đầu tư); trả nợ vay nước ngoài hoặc khoản cấp "
                "tín dụng tại TCTD khác (trừ cho vay trả nợ trước hạn đủ điều kiện); "
                "gửi tiền; góp vốn, mua, nhận chuyển nhượng phần vốn góp, cổ phần chưa "
                "niêm yết; dự án đầu tư không đủ điều kiện đưa vào kinh doanh (Điều 8 "
                "TT 39/2016/TT-NHNN; khoản 8, 9 ngưng hiệu lực theo TT 10/2023/TT-NHNN)."
            ),
            "author": NOTE_TT39,
        },
    ),
    "52": (
        {"doc_link": U_TT06_2023, "passage_prefix": "Điều 13: Lãi suất VND do thỏa thuận"},
        {
            "doc_link": U_VBHN_39,
            "text_contains_answer_in_the_doc": (
                "Điều 13: Tổ chức tín dụng và khách hàng thỏa thuận về lãi suất cho vay "
                "theo cung cầu vốn thị trường, nhu cầu vay vốn và mức độ tín nhiệm của "
                "khách hàng (khoản 1); lãi suất cho vay ngắn hạn bằng đồng Việt Nam "
                "phục vụ một số nhu cầu vốn không vượt quá mức tối đa do Thống đốc "
                "NHNN quyết định, áp dụng khi khách hàng được đánh giá có tình hình tài "
                "chính minh bạch, lành mạnh (khoản 2, sửa đổi bởi TT 06/2023/TT-NHNN)."
            ),
            "answer": (
                "Lãi suất cho vay bằng VND do TCTD và khách hàng thỏa thuận theo cung "
                "cầu vốn thị trường, nhu cầu vay vốn và mức độ tín nhiệm của khách hàng "
                "(khoản 1 Điều 13 TT 39/2016). Cho vay ngắn hạn bằng VND phục vụ một số "
                "nhu cầu vốn (nông nghiệp, nông thôn; xuất khẩu; doanh nghiệp nhỏ và "
                "vừa; công nghiệp hỗ trợ; công nghệ cao) không vượt mức lãi suất tối đa "
                "do Thống đốc NHNN quyết định trong từng thời kỳ, chỉ áp dụng với khách "
                "hàng được TCTD đánh giá có tình hình tài chính minh bạch, lành mạnh "
                "(khoản 2, sửa đổi bởi TT 06/2023)."
            ),
            "author": NOTE_TT39,
        },
    ),
    "53": (
        {"doc_link": U_TT06_2023, "passage_prefix": "Điều 7: Điều kiện vay: năng lực pháp luật"},
        {
            "doc_link": U_VBHN_39,
            "text_contains_answer_in_the_doc": (
                "Điều 7: Tổ chức tín dụng xem xét, quyết định cho vay khi khách hàng có "
                "đủ các điều kiện: khách hàng là pháp nhân có năng lực pháp luật dân "
                "sự, cá nhân từ đủ 18 tuổi có năng lực hành vi dân sự đầy đủ hoặc từ "
                "đủ 15 đến dưới 18 tuổi không bị mất, hạn chế năng lực hành vi dân sự "
                "(khoản 1); nhu cầu vay vốn sử dụng vào mục đích hợp pháp (khoản 2); "
                "phương án sử dụng vốn khả thi (khoản 3); khả năng tài chính để trả nợ "
                "(khoản 4); khoản 5 đã được bãi bỏ bởi TT 06/2023."
            ),
            "answer": (
                "Khách hàng phải có đủ: (1) năng lực pháp luật dân sự (pháp nhân) hoặc "
                "năng lực hành vi dân sự đầy đủ - cá nhân từ đủ 18 tuổi, hoặc từ đủ 15 "
                "đến dưới 18 tuổi không bị mất hoặc hạn chế năng lực hành vi dân sự; "
                "(2) nhu cầu vay vốn sử dụng vào mục đích hợp pháp; (3) phương án sử "
                "dụng vốn khả thi, trừ khoản cho vay có mức giá trị nhỏ; (4) khả năng "
                "tài chính để trả nợ (Điều 7 TT 39/2016/TT-NHNN; khoản 5 đã bị bãi bỏ "
                "bởi TT 06/2023)."
            ),
            "author": NOTE_TT39,
        },
    ),
    "54": (
        {"doc_link": U_TT06_2023, "passage_prefix": "Điều 21: Kiểm tra giám sát đúng mục đích"},
        {
            "doc_link": U_VBHN_39,
            "text_contains_answer_in_the_doc": (
                "Điều 24: Khách hàng có nghĩa vụ sử dụng vốn vay đúng mục đích đã cam "
                "kết, hoàn trả nợ gốc, lãi, phí đầy đủ, đúng hạn, báo cáo việc sử dụng "
                "vốn vay (khoản 1); tổ chức tín dụng có quyền, nghĩa vụ kiểm tra, giám "
                "sát việc sử dụng vốn vay và trả nợ của khách hàng (khoản 2; Điều 24 "
                "được sửa đổi bởi TT 06/2023, thay thế bởi TT 12/2024)."
            ),
            "answer": (
                "Khách hàng có nghĩa vụ sử dụng vốn vay đúng mục đích đã cam kết, hoàn "
                "trả nợ gốc, lãi, phí đúng hạn và cung cấp thông tin, tài liệu, dữ liệu "
                "chứng minh; TCTD có quyền, nghĩa vụ kiểm tra, giám sát việc sử dụng "
                "vốn vay và trả nợ của khách hàng (khoản 1, khoản 2 Điều 24 TT "
                "39/2016, sửa đổi bởi TT 06/2023, TT 12/2024). Khi phát hiện khách hàng "
                "cung cấp thông tin sai sự thật hoặc vi phạm thỏa thuận cho vay, TCTD "
                "có quyền chấm dứt cho vay, thu hồi nợ trước hạn (khoản 1 Điều 21)."
            ),
            "author": NOTE_TT39,
        },
    ),
    "55": (
        {"doc_link": U_TT06_2023, "passage_prefix": "Điều 1 Khoản 8: Cho vay điện tử"},
        {
            "doc_link": U_VBHN_39,
            "text_contains_answer_in_the_doc": (
                "Điều 32a: Tổ chức tín dụng thực hiện cho vay bằng phương tiện điện tử "
                "phù hợp với điều kiện hoạt động kinh doanh, đảm bảo an ninh, an toàn, "
                "bảo mật thông tin; phải có giải pháp nhận biết, xác minh thông tin "
                "nhận biết khách hàng (Điều 32b) - Mục 3 Chương II TT 39/2016 bổ sung "
                "bởi TT 06/2023."
            ),
            "answer": (
                "Cho vay bằng phương tiện điện tử áp dụng đối với khách hàng là cá nhân "
                "vay vốn phục vụ nhu cầu đời sống (khoản 2 Điều 32b, Điều 32c TT "
                "39/2016, Mục 3 bổ sung bởi TT 06/2023). TCTD phải có giải pháp, công "
                "nghệ nhận biết, xác minh thông tin khách hàng, kể cả dữ liệu sinh trắc "
                "học, đảm bảo khách hàng giao dịch chính là khách hàng vay vốn (Điều "
                "32b); thẩm định, quyết định cho vay trên hồ sơ điện tử (Điều 32d, Điều "
                "32đ); lưu trữ, bảo quản thông tin, dữ liệu bảo đảm an toàn, bảo mật và "
                "được sao lưu dự phòng (khoản 3 Điều 32a)."
            ),
            "author": NOTE_TT39,
        },
    ),
    "56": (
        {"doc_link": U_TT06_2023, "passage_prefix": "Điều 18: Giải ngân chuyển khoản"},
        {
            "doc_link": U_TT21_2017,
            "text_contains_answer_in_the_doc": (
                "Điều 4: Tổ chức tín dụng cho vay phải sử dụng dịch vụ thanh toán không "
                "dùng tiền mặt theo quy định của pháp luật để giải ngân vốn cho vay vào "
                "tài khoản thanh toán của bên thụ hưởng tại tổ chức cung ứng dịch vụ "
                "thanh toán, trừ trường hợp quy định tại khoản 2 Điều này, Điều 5 và "
                "Điều 6 Thông tư này (TT 21/2017/TT-NHNN)."
            ),
            "answer": (
                "Về nguyên tắc giải ngân phải sử dụng dịch vụ thanh toán không dùng tiền "
                "mặt vào tài khoản thanh toán của bên thụ hưởng (Điều 4 TT "
                "21/2017/TT-NHNN). Giải ngân bằng tiền mặt chỉ được xem xét quyết định "
                "khi bên thụ hưởng (không bao gồm pháp nhân) không có tài khoản thanh "
                "toán - khách hàng thanh toán, chi trả cho bên thụ hưởng không có tài "
                "khoản, hoặc khách hàng là bên thụ hưởng đã ứng vốn tự có - và khách "
                "hàng phải gửi văn bản cam kết của bên thụ hưởng (Điều 5). Với khoản "
                "vay không quá 100 triệu đồng trả cho bên thụ hưởng có tài khoản "
                "(không bao gồm pháp nhân), hoặc bên thụ hưởng là tổ chức sử dụng vốn "
                "nhà nước được thanh toán bằng tiền mặt, TCTD được quyết định phương "
                "thức phù hợp (Điều 6)."
            ),
            "author": NOTE_TT39,
        },
    ),
    "57": (
        {"doc_link": U_TT06_2023, "passage_prefix": "Điều 10: Thời hạn cho vay dựa trên"},
        {
            "doc_link": U_VBHN_39,
            "text_contains_answer_in_the_doc": (
                "Điều 28: Tổ chức tín dụng và khách hàng căn cứ vào chu kỳ hoạt động "
                "kinh doanh, thời hạn thu hồi vốn, khả năng trả nợ của khách hàng, "
                "nguồn vốn cho vay và thời hạn hoạt động còn lại của tổ chức tín dụng "
                "để thỏa thuận về thời hạn cho vay."
            ),
            "answer": (
                "Thời hạn cho vay do TCTD và khách hàng thỏa thuận trên cơ sở: chu kỳ "
                "hoạt động kinh doanh, thời hạn thu hồi vốn, khả năng trả nợ của khách "
                "hàng, nguồn vốn cho vay và thời hạn hoạt động còn lại của TCTD (khoản "
                "1 Điều 28 TT 39/2016; với cho vay phục vụ nhu cầu đời sống dùng cơ sở "
                "tương ứng tại Điều 31)."
            ),
            "author": NOTE_TT39,
        },
    ),
    "58": (
        {"doc_link": U_TT06_2023, "passage_prefix": "Điều 14: Chuyển nợ quá hạn khi không"},
        {
            "doc_link": U_VBHN_39,
            "text_contains_answer_in_the_doc": (
                "Điều 20: Tổ chức tín dụng chuyển nợ quá hạn đối với số dư nợ gốc mà "
                "khách hàng không trả được nợ đúng hạn theo thỏa thuận và không được "
                "tổ chức tín dụng chấp thuận cơ cấu lại thời hạn trả nợ; thông báo cho "
                "khách hàng về số dư nợ gốc bị quá hạn, thời điểm chuyển nợ quá hạn, "
                "lãi suất áp dụng. Lãi suất trên dư nợ gốc bị quá hạn không vượt quá "
                "150% lãi suất cho vay trong hạn (Điều 13 khoản 4 điểm c)."
            ),
            "answer": (
                "TCTD chuyển nợ quá hạn đối với số dư nợ gốc mà khách hàng không trả "
                "được đúng hạn theo thỏa thuận cho vay và không được TCTD chấp thuận cơ "
                "cấu lại thời hạn trả nợ, đồng thời thông báo cho khách hàng về số dư "
                "nợ gốc bị quá hạn, thời điểm chuyển nợ quá hạn và lãi suất áp dụng "
                "(Điều 20 TT 39/2016). Lãi suất trên dư nợ gốc bị quá hạn không vượt "
                "quá 150% lãi suất cho vay trong hạn tại thời điểm chuyển nợ quá hạn "
                "(khoản 4 điểm c Điều 13)."
            ),
            "author": NOTE_TT39,
        },
    ),
    "65": (
        {
            "doc_link_prefix": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2019/01/49-nhnn.pdf",
            "answer_contains": "Thông tư 49/2018/TT-NHNN quy định về tiền gửi có kỳ hạn",
        },
        {
            "answer": None,  # built in main(): two instrument-number fixes on the memo text
            "author": (
                "Thành; corrected 2026-09-11: Thông tư 49/2018 và 49/2019 không tồn tại - "
                "file 49-nhnn.pdf là TT 22/2018/TT-NHNN (tiền gửi có kỳ hạn), xác minh bằng "
                "OCR header; chỉ sửa số hiệu văn bản, nội dung tư vấn giữ nguyên"
            ),
        },
    ),
}

# Answer rows (post-correction) that keep their original answer verbatim:
# 29, 31, 36 - only the locator/doc_link/author change.


def load_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    return header, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    header, rows = load_rows(args.csv)
    idx = {name: i for i, name in enumerate(header)}
    need = ["ID", "doc_link", "question", "answer", "text_contains_answer_in_the_doc", "author"]
    missing = [c for c in need if c not in idx]
    if missing:
        raise SystemExit(f"CSV missing column(s): {missing}")

    by_id = {}
    for pos, row in enumerate(rows, 1):
        rid = row[idx["ID"]].strip()
        if rid:
            if rid in by_id:
                raise SystemExit(f"duplicate explicit ID {rid!r} in CSV")
            by_id[rid] = (pos, row)

    errors, applied, skipped_done = [], [], []

    for rid, (expected, after) in sorted(CORRECTIONS.items(), key=lambda kv: int(kv[0])):
        if rid not in by_id:
            errors.append(f"row ID {rid} not found")
            continue
        pos, row = by_id[rid]

        # Q065: the corrected answer is the current memo text with the two
        # non-existent instrument numbers replaced; build it before guarding.
        after = dict(after)
        if after.get("answer") is None:
            after["answer"] = row[idx["answer"]].replace(
                "Thông tư 49/2018/TT-NHNN quy định về tiền gửi có kỳ hạn",
                "Thông tư 22/2018/TT-NHNN quy định về tiền gửi có kỳ hạn",
            ).replace("Thông tư 49/2019/TT-NHNN", "Thông tư 22/2018/TT-NHNN")

        # Idempotency: a row already carrying every after-value is done.
        if all(row[idx[col]] == value for col, value in after.items()):
            skipped_done.append(rid)
            continue

        # A guard passes on EITHER the original cell state or a previously
        # applied after-value, so a correction entry can be amended in place
        # (e.g. when new evidence fixes a number in an already-corrected row).
        if "doc_link" in expected:
            allowed = {expected["doc_link"]}
            if "doc_link" in after:
                allowed.add(after["doc_link"])
            if row[idx["doc_link"]] not in allowed:
                errors.append(
                    f"ID {rid}: doc_link is {row[idx['doc_link']]!r}, "
                    f"expected one of {sorted(allowed)!r}"
                )
                continue
        if "doc_link_prefix" in expected and not row[idx["doc_link"]].startswith(
            expected["doc_link_prefix"]
        ):
            errors.append(
                f"ID {rid}: doc_link starts {row[idx['doc_link']][:60]!r}, "
                f"expected prefix {expected['doc_link_prefix'][:60]!r}"
            )
            continue
        if "answer_contains" in expected and expected["answer_contains"] not in row[idx["answer"]]:
            errors.append(
                f"ID {rid}: answer lacks {expected['answer_contains'][:60]!r} - "
                "the row it was written against has moved"
            )
            continue
        prefixes = [p for p in (expected.get("passage_prefix"), after.get("text_contains_answer_in_the_doc")) if p]
        if prefixes and not any(
            row[idx["text_contains_answer_in_the_doc"]].startswith(p) for p in prefixes
        ):
            errors.append(
                f"ID {rid}: passage starts {row[idx['text_contains_answer_in_the_doc']][:60]!r}, "
                f"expected one of the recorded prefixes"
            )
            continue
        for col, value in after.items():
            row[idx[col]] = value
        applied.append(rid)

    # Pin explicit ids on the trailing probe rows (blank ID cells at positions 66+).
    pinned = []
    for pos, row in enumerate(rows, 1):
        if row[idx["ID"]].strip() == "" and (row[idx["question"]] or "").strip():
            if pos < 66:
                errors.append(f"blank ID on non-tail row {pos} - refusing to guess")
                continue
            row[idx["ID"]] = str(pos)
            pinned.append(pos)

    if errors:
        print("REFUSING to write - guards failed:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print(f"corrections applied to {len(applied)} row(s): {', '.join(applied) or '-'}")
    if skipped_done:
        print(f"already applied (skipped): {', '.join(skipped_done)}")
    print(f"probe ids pinned on {len(pinned)} row(s): {pinned[0]}..{pinned[-1]}" if pinned
          else "no probe rows needed pinning")

    if args.dry_run:
        print("[dry-run] CSV not modified")
        return 0

    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\r\n")
        writer.writerow(header)
        writer.writerows(rows)
    print(f"wrote {os.path.relpath(args.csv, REPO_ROOT)}")
    print("now re-run: scripts/update_doc_manifest.py && scripts/load_qa_csv.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
