# gen_handshake_cost.py
# Airtime and computation of one AP-initiated 4-way handshake (PTK rekey),
# 5 GHz OFDM (Sec. IV-D). Analytic, no dataset.
#
#   python -m experiments.gen_handshake_cost
import math, os, hmac, hashlib, time

# EAPOL-Key frame = QoS-data MAC hdr 26 + CCMP hdr 8 + LLC/SNAP 8 + EAPOL hdr 4
#                   + EAPOL-Key body 95 + key data + CCMP MIC 8 + FCS 4
base = 26 + 8 + 8 + 4 + 95 + 8 + 4
keydata = {"M1": 0, "M2": 22, "M3": 22 + 24 + 30 + 8 + 4, "M4": 0}   # RSNE, GTK/IGTK KDE, wrap
frames = {k: base + v for k, v in keydata.items()}
SIFS, SLOT, DIFS = 16, 9, 34
backoff = 7.5 * SLOT                                     # mean of CWmin=15

def ppdu(nbytes, mbps):
    ndbps = 4 * mbps                                     # bits per 4-us OFDM symbol
    return 20 + 4 * math.ceil((16 + 8 * nbytes + 6) / ndbps)

for r in (6, 24):
    per = [DIFS + backoff + ppdu(b, r) + SIFS + ppdu(14, r) for b in frames.values()]
    print(f"{r:>2} Mb/s: frames {frames}  airtime {sum(per)/1000:.2f} ms")

# computation: PTK derivation (PRF-384, HMAC-SHA1 x3) + MIC on each of 4 frames
pmk, a, s = os.urandom(32), os.urandom(32), os.urandom(32)
def prf(key, label, data, nbits=384):
    out = b""
    for i in range(math.ceil(nbits / 160)):
        out += hmac.new(key, label + b"\x00" + data + bytes([i]), hashlib.sha1).digest()
    return out[: nbits // 8]
def one():
    ptk = prf(pmk, b"Pairwise key expansion", os.urandom(12) + a + s)
    for b in frames.values():
        hmac.new(ptk[:16], os.urandom(b), hashlib.sha1).digest()
n = 20000
t = time.perf_counter()
for _ in range(n): one()
print(f"crypto per handshake (PTK + 4 MICs, python): {(time.perf_counter()-t)/n*1e6:.1f} us")
