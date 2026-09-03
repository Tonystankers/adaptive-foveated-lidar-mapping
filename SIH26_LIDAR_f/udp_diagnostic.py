"""
udp_diagnostic.py - Bare-bones UDP connectivity test

Has ZERO dependency on Streamlit, numpy, or anything else in this project.
Its only job is to answer one question: can two plain Python processes on
this machine talk to each other over UDP at all? If this fails, the problem
is Windows Firewall / antivirus / network config - not the dashboard code.
If this works but the dashboard still doesn't, the problem is narrowed down
to something in the Streamlit/threading integration.

Usage (two separate terminals, same machine):

    Terminal A:  python udp_diagnostic.py listen
    Terminal B:  python udp_diagnostic.py send

Terminal A should immediately start printing "Received packet #N" lines.
"""

import socket
import sys
import time

HOST = "127.0.0.1"
PORT = 5599


def listen():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((HOST, PORT))
    except OSError as exc:
        print(f"[FAIL] Could not bind to {HOST}:{PORT} -> {exc}")
        print("       This points to the port being in use, or a firewall/permissions block.")
        sys.exit(1)

    print(f"[OK] Bound and listening on {HOST}:{PORT}. Waiting for packets... (Ctrl+C to stop)")
    count = 0
    while True:
        try:
            data, addr = sock.recvfrom(4096)
            count += 1
            print(f"[OK] Received packet #{count} from {addr}: {data!r}")
        except KeyboardInterrupt:
            print("\nStopped.")
            break


def send():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"[*] Sending 10 test packets to {HOST}:{PORT} ...")
    for i in range(10):
        msg = f"test-packet-{i}".encode("utf-8")
        sock.sendto(msg, (HOST, PORT))
        print(f"    sent: {msg!r}")
        time.sleep(0.5)
    print("[*] Done. If the listener didn't print anything, packets are being dropped")
    print("    somewhere between the two processes (firewall/AV is the usual cause).")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("listen", "send"):
        print("Usage: python udp_diagnostic.py [listen|send]")
        sys.exit(1)
    if sys.argv[1] == "listen":
        listen()
    else:
        send()
