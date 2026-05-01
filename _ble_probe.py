"""One-shot BLE scan to verify T1/A1 are advertising the v3 service.

Independent of uwb_dashboard.py — just lists nearby devices, then for any
device named T1 or A1, connects and dumps its services/characteristics so
we can confirm the v3 BLE characteristic UUIDs are present.
"""
import asyncio
from bleak import BleakScanner, BleakClient

UWB_SERVICE_UUID = "19b10010-e8f2-537e-4f6c-d104768a1214"
EXPECTED_CHARS = {
    "19b10011-e8f2-537e-4f6c-d104768a1214": "TAG_CHAR (T1)",
    "19b10012-e8f2-537e-4f6c-d104768a1214": "ANCHOR_CHAR (A1)",
    "19b10013-e8f2-537e-4f6c-d104768a1214": "CMD_CHAR",
}


async def main():
    print("[scan] 6s passive scan…")
    found = await BleakScanner.discover(timeout=6.0, return_adv=True)
    matches = []
    for addr, (dev, adv) in found.items():
        name = (dev.name or adv.local_name or "").strip()
        if name in ("T1", "A1"):
            matches.append((name, addr, dev, adv))
            print(f"  [+] {name}  addr={addr}  rssi={adv.rssi}  service_uuids={adv.service_uuids}")
    if not matches:
        print("  [-] No T1/A1 advertising. Boards either not powered, not flashed, or BLE not started.")
        names = sorted({(d.name or a.local_name or "?") for _, (d, a) in found.items()})
        print(f"  Visible names this scan: {names[:15]}")
        return

    for name, addr, dev, adv in matches:
        print(f"\n[connect] {name} @ {addr}")
        try:
            async with BleakClient(addr, timeout=10.0) as client:
                print(f"  connected={client.is_connected}")
                services = client.services
                for svc in services:
                    print(f"  service {svc.uuid}")
                    for ch in svc.characteristics:
                        tag = EXPECTED_CHARS.get(ch.uuid.lower(), "")
                        print(f"    char {ch.uuid}  props={','.join(ch.properties)}  {tag}")
        except Exception as e:
            print(f"  [!] connect failed: {e!r}")


if __name__ == "__main__":
    asyncio.run(main())
