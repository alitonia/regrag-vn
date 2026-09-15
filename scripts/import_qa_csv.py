"""DEPRECATED: Old importer that produced broken gold citations with UNKNOWN doc_ids.

Do not use this script. Use scripts/load_qa_csv.py instead.
"""

import sys


def main() -> None:
    print(
        "ERROR: scripts/import_qa_csv.py is DEPRECATED and MUST NOT be used.\n"
        "Its gold-citation derivation regexed doc_id from answer prose, which failed\n"
        "because only 2 of 64 gold passages name their own legal instrument (producing\n"
        "83 UNKNOWN doc_ids and making citation accuracy unscorable).\n"
        "\nPlease use 'scripts/load_qa_csv.py' instead, which resolves canonical doc_ids\n"
        "from the doc_link column via DOC_MANIFEST.json and canonical URL slugs.",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()

