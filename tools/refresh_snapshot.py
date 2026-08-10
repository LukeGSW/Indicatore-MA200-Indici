"""
refresh_snapshot.py — Rigenera src/constituents_snapshot.py dalle fonti online.

Lo snapshot è l'ultima rete di sicurezza della catena di fonti: entra in gioco
solo quando TUTTE le fonti online hanno fallito, e serve a non far morire la
dashboard per un problema di rete o per la rotazione di un endpoint.

Va rigenerato ogni tanto (indicativamente ogni trimestre, dopo i ribilanciamenti
di indice) — non è automatico di proposito: un file versionato e datato dice a
colpo d'occhio quanto è vecchia la rete di sicurezza.

Uso, dalla radice del repository:

    python -m tools.refresh_snapshot

Lo script accetta una lista solo se almeno DUE fonti indipendenti concordano:
congelare nel repo la lista di una singola fonte non verificata significherebbe
propagare per mesi un eventuale errore.
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import constituents as C          # noqa: E402

OUTPUT = Path(__file__).resolve().parent.parent / "src" / "constituents_snapshot.py"

HEADER = '''"""
constituents_snapshot.py — Lista di costituenti congelata, usata come ultima
risorsa quando tutte le fonti online falliscono.

NON modificare a mano: rigenerare con `python -m tools.refresh_snapshot`.
Ogni lista qui sotto è stata accettata solo dopo la concordanza di almeno due
fonti indipendenti alla data indicata.

Generato il {today}.
"""

from datetime import date

SNAPSHOT_DATE = date({y}, {m}, {d})

SNAPSHOTS: dict[str, list[str]] = {{
'''


def main() -> int:
    today = date.today()
    blocks: list[str] = []
    failed: list[str] = []

    for index_key, (market, providers) in C._PROVIDERS.items():
        lists: dict[str, set[str]] = {}

        for label, provider in providers:
            try:
                symbols, _ = provider()
                tickers, _ = C._to_eodhd(symbols, market)
                C._check_plausible(index_key, tickers)
                lists[label] = set(tickers)
                print(f"  [ok]     {index_key:7s} {label:42s} n={len(tickers)}")
            except Exception as e:                                # noqa: BLE001
                print(f"  [scarto] {index_key:7s} {label:42s} {type(e).__name__}: {e}")

        if len(lists) < 2:
            failed.append(f"{index_key}: solo {len(lists)} fonte/i valida/e, ne servono 2")
            continue

        # Si tiene la lista su cui converge il maggior numero di fonti.
        by_signature: dict[tuple[str, ...], list[str]] = {}
        for label, tickers in lists.items():
            by_signature.setdefault(tuple(sorted(tickers)), []).append(label)

        signature, agreeing = max(by_signature.items(), key=lambda kv: len(kv[1]))

        if len(agreeing) < 2:
            failed.append(
                f"{index_key}: nessuna coppia di fonti concordi "
                f"({len(by_signature)} liste diverse da {len(lists)} fonti)"
            )
            continue

        if len(by_signature) > 1:
            print(f"  [avviso] {index_key}: fonti discordi, uso quella condivisa "
                  f"da {len(agreeing)}/{len(lists)} ({', '.join(agreeing)})")

        entries = "\n".join(f'        "{t}",' for t in signature)
        blocks.append(
            f'    # {index_key} — {len(signature)} costituenti, concordi su '
            f'{len(agreeing)}/{len(lists)} fonti\n'
            f'    "{index_key}": [\n{entries}\n    ],'
        )

    if failed:
        print("\nERRORE — snapshot NON scritto:")
        for f in failed:
            print(f"  - {f}")
        return 1

    OUTPUT.write_text(
        HEADER.format(today=today.strftime("%d/%m/%Y"), y=today.year,
                      m=today.month, d=today.day)
        + "\n".join(blocks)
        + "\n}\n",
        encoding="utf-8",
    )
    print(f"\nScritto {OUTPUT} ({today:%d/%m/%Y})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
