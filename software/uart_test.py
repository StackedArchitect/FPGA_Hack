#!/usr/bin/env python3
"""UART test: send ECG beats to FPGA, receive classification results."""

import argparse
import os
import sys
import time
import struct
import numpy as np

# ── Configuration ──
NUM_SAMPLES   = 187
BAUD_RATE     = 115_200
AAMI_CLASSES  = ["N", "S", "V", "F", "Q"]
AAMI_LABELS   = {
    0: "N  (Normal)",
    1: "S  (Supraventricular)",
    2: "V  (Ventricular)",
    3: "F  (Fusion)",
    4: "Q  (Unknown/Paced)",
}
BIT_PERIOD_S  = 1.0 / BAUD_RATE       # ~8.68 µs
BYTE_PERIOD_S = 10 * BIT_PERIOD_S      # start + 8 data + stop
TX_TIME_S     = NUM_SAMPLES * BYTE_PERIOD_S  # ~16.2 ms for 187 bytes
# Inference takes ~653 cycles @ 100 MHz = ~6.5 µs
# Total round-trip: ~16.2 ms TX + ~6.5 µs inference + ~86.8 µs RX = ~16.4 ms
RX_TIMEOUT_S  = 2.0  # generous timeout for result byte


def quantize_for_fpga(X: np.ndarray, bits: int = 8) -> np.ndarray:
    """Quantize float32 beats to signed int8 (maps ±3σ → ±127)."""
    max_val = 2 ** (bits - 1) - 1  # 127
    scale = max_val / 3.0
    return np.clip(np.round(X * scale), -max_val - 1, max_val).astype(np.int8)


def load_test_data_csv(csv_path: str) -> tuple:
    """Load test data from Kaggle MIT-BIH CSV (N × 188: 187 samples + label)."""
    print(f"  Loading: {csv_path}")
    data = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
    X = data[:, :-1]   # (N, 187) float32
    y = data[:, -1].astype(int)
    X_q = quantize_for_fpga(X)
    print(f"  Loaded {len(X)} beats ({X.shape[1]} samples each)")
    return X_q, y


def load_test_data_mem(mem_path: str, label_path: str = None) -> tuple:
    """Load test data from exported .mem files (hex format)."""
    print(f"  Loading: {mem_path}")
    beats = []
    with open(mem_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            hex_vals = line.split()
            byte_vals = [int(v, 16) for v in hex_vals]
            # Convert unsigned byte back to signed int8
            signed_vals = [v if v < 128 else v - 256 for v in byte_vals]
            beats.append(signed_vals[:NUM_SAMPLES])

    X_q = np.array(beats, dtype=np.int8)
    y = None
    if label_path and os.path.exists(label_path):
        y = np.loadtxt(label_path, dtype=int)
    print(f"  Loaded {len(X_q)} beats")
    return X_q, y


def send_beat(ser, samples: np.ndarray) -> int:
    """
    Send one heartbeat (187 int8 samples) and receive classification result.

    Returns:
        Predicted class (0-4), or -1 on timeout.
    """
    # Convert signed int8 → unsigned bytes for UART
    tx_bytes = bytes([s & 0xFF for s in samples.astype(np.int8)])
    assert len(tx_bytes) == NUM_SAMPLES, f"Expected {NUM_SAMPLES} bytes, got {len(tx_bytes)}"

    # Flush input buffer
    ser.reset_input_buffer()

    # Send all samples
    ser.write(tx_bytes)
    ser.flush()

    # Wait for 1-byte result with timeout
    ser.timeout = RX_TIMEOUT_S
    result = ser.read(1)

    if len(result) == 0:
        return -1  # timeout

    return result[0] & 0x07  # bits [2:0]


def run_test(ser, X_q: np.ndarray, y: np.ndarray, num_tests: int = None,
             start_idx: int = 0) -> dict:
    """Run classification tests and report results."""
    if num_tests is None:
        num_tests = len(X_q) - start_idx
    num_tests = min(num_tests, len(X_q) - start_idx)

    correct = 0
    total = 0
    class_correct = [0] * 5
    class_total = [0] * 5
    results = []

    print(f"\n{'='*65}")
    print(f"  Running {num_tests} test vectors (starting at index {start_idx})")
    print(f"  Port: {ser.port} @ {ser.baudrate} baud")
    print(f"{'='*65}\n")

    t_start = time.time()

    for i in range(num_tests):
        idx = start_idx + i
        samples = X_q[idx]
        expected = y[idx] if y is not None else None

        t0 = time.time()
        predicted = send_beat(ser, samples)
        elapsed_ms = (time.time() - t0) * 1000

        total += 1

        if predicted < 0:
            status = "TIMEOUT"
        elif expected is not None:
            if predicted == expected:
                correct += 1
                class_correct[expected] += 1
                status = "PASS"
            else:
                status = "FAIL"
            class_total[expected] += 1
        else:
            status = f"class={predicted}"

        # Print progress
        exp_str = AAMI_LABELS.get(expected, "?") if expected is not None else "?"
        pred_str = AAMI_LABELS.get(predicted, "TIMEOUT")
        if i < 20 or i % 50 == 0 or status in ("FAIL", "TIMEOUT") or i == num_tests - 1:
            print(f"  [{i+1:4d}/{num_tests}] Expected: {expected} ({AAMI_CLASSES[expected] if expected is not None else '?'})"
                  f"  Got: {predicted} ({AAMI_CLASSES[predicted] if predicted >= 0 else '?'})"
                  f"  {status}  ({elapsed_ms:.1f} ms)")

        results.append({
            "index": idx,
            "expected": expected,
            "predicted": predicted,
            "status": status,
            "time_ms": elapsed_ms,
        })

    total_time = time.time() - t_start

    # ── Summary ──
    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")
    if y is not None:
        accuracy = correct / total if total > 0 else 0
        print(f"  Accuracy:   {correct}/{total} = {100*accuracy:.2f}%")
        print(f"  Total time: {total_time:.2f}s ({total_time/total*1000:.1f} ms/beat)")
        print()
        print(f"  {'Class':<25s} {'Correct':>8s} {'Total':>8s} {'Accuracy':>10s}")
        print(f"  {'-'*51}")
        for c in range(5):
            if class_total[c] > 0:
                acc = class_correct[c] / class_total[c]
                print(f"  {AAMI_LABELS[c]:<25s} {class_correct[c]:>8d} {class_total[c]:>8d} {100*acc:>9.2f}%")
    else:
        print(f"  Processed {total} beats in {total_time:.2f}s")

    print(f"{'='*65}\n")
    return {"results": results, "correct": correct, "total": total}


def interactive_mode(ser, X_q, y):
    """Interactive menu for sending individual beats."""
    print(f"\n  Interactive mode — {len(X_q)} beats loaded")
    print(f"  Commands: <index>, 'r' (random), 'q' (quit)\n")

    while True:
        try:
            cmd = input("  Beat index> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd in ("q", "quit", "exit"):
            break
        elif cmd in ("r", "random"):
            idx = np.random.randint(len(X_q))
        else:
            try:
                idx = int(cmd)
            except ValueError:
                print("  Invalid input. Enter a number, 'r', or 'q'.")
                continue

        if idx < 0 or idx >= len(X_q):
            print(f"  Index out of range (0-{len(X_q)-1})")
            continue

        expected = y[idx] if y is not None else None
        exp_str = f"{expected} ({AAMI_LABELS[expected]})" if expected is not None else "unknown"
        print(f"  Sending beat #{idx} (expected: {exp_str})...")

        t0 = time.time()
        predicted = send_beat(ser, X_q[idx])
        elapsed_ms = (time.time() - t0) * 1000

        if predicted < 0:
            print(f"  Result: TIMEOUT ({elapsed_ms:.1f} ms)")
        else:
            pred_str = f"{predicted} ({AAMI_LABELS[predicted]})"
            match = "✓ PASS" if (expected is not None and predicted == expected) else \
                    ("✗ FAIL" if expected is not None else "")
            print(f"  Result: {pred_str}  {match}  ({elapsed_ms:.1f} ms)")
        print()


def loopback_test(ser):
    """Quick loopback test — send a byte and check if it echoes (for wiring check)."""
    print("  Loopback test: short TX to RX, send 0xAA...")
    ser.reset_input_buffer()
    ser.write(b"\xAA")
    ser.flush()
    ser.timeout = 0.5
    result = ser.read(1)
    if result:
        print(f"  Received: 0x{result[0]:02X} — {'PASS' if result[0] == 0xAA else 'UNEXPECTED'}")
    else:
        print("  No echo received (normal if connected to FPGA, not loopback)")


def main():
    parser = argparse.ArgumentParser(
        description="WaveBNN-ECG UART Test: send ECG beats to FPGA, receive classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python uart_test.py --port /dev/ttyUSB0 --test 10
  python uart_test.py --port COM3 --csv data/mitbih_test.csv --test 100
  python uart_test.py --port /dev/ttyUSB0 --interactive
  python uart_test.py --port /dev/ttyUSB0 --mem ../hardware/tb/test_vectors/test_input.mem
        """,
    )
    parser.add_argument("--port", type=str, default=None,
                        help="Serial port (e.g. /dev/ttyUSB0 or COM3)")
    parser.add_argument("--baud", type=int, default=BAUD_RATE,
                        help=f"Baud rate (default: {BAUD_RATE})")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to Kaggle mitbih_test.csv")
    parser.add_argument("--mem", type=str, default=None,
                        help="Path to test_input.mem file")
    parser.add_argument("--labels", type=str, default=None,
                        help="Path to test_labels.mem file (used with --mem)")
    parser.add_argument("--test", type=int, default=None,
                        help="Number of test vectors to run")
    parser.add_argument("--start", type=int, default=0,
                        help="Starting index in test data")
    parser.add_argument("--index", type=int, default=None,
                        help="Send a single beat at this index")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode: choose beats to send")
    parser.add_argument("--loopback", action="store_true",
                        help="Quick loopback wiring test")
    parser.add_argument("--list-ports", action="store_true",
                        help="List available serial ports")
    args = parser.parse_args()

    # ── List ports ──
    if args.list_ports:
        try:
            from serial.tools.list_ports import comports
            ports = list(comports())
            if ports:
                print("Available serial ports:")
                for p in ports:
                    print(f"  {p.device:20s}  {p.description}")
            else:
                print("No serial ports found.")
        except ImportError:
            print("Install pyserial: pip install pyserial")
        return

    # ── Import pyserial ──
    try:
        import serial
    except ImportError:
        print("ERROR: pyserial is required.")
        print("Install it with: pip install pyserial")
        sys.exit(1)

    # ── Auto-detect data source ──
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.csv:
        data_path = args.csv
    elif args.mem:
        data_path = args.mem
    else:
        # Try default locations
        csv_default = os.path.join(script_dir, "data", "mitbih_test.csv")
        mem_default = os.path.join(script_dir, "..", "hardware", "tb", "test_vectors", "test_input.mem")
        if os.path.exists(csv_default):
            data_path = csv_default
            args.csv = csv_default
        elif os.path.exists(mem_default):
            data_path = mem_default
            args.mem = mem_default
        else:
            print("ERROR: No test data found.")
            print(f"  Searched: {csv_default}")
            print(f"           {mem_default}")
            print("  Use --csv or --mem to specify data path.")
            sys.exit(1)

    # ── Load test data ──
    print(f"\n{'='*65}")
    print(f"  WaveBNN-ECG UART Test")
    print(f"{'='*65}")

    if args.csv:
        X_q, y = load_test_data_csv(data_path)
    else:
        label_path = args.labels
        if label_path is None:
            auto_label = data_path.replace("test_input", "test_labels")
            if os.path.exists(auto_label):
                label_path = auto_label
        X_q, y = load_test_data_mem(data_path, label_path)

    # ── Open serial port ──
    if args.port is None:
        # Try auto-detect
        try:
            from serial.tools.list_ports import comports
            ports = [p.device for p in comports()]
            if len(ports) == 1:
                args.port = ports[0]
                print(f"  Auto-detected port: {args.port}")
            elif len(ports) > 1:
                print(f"  Multiple ports found: {ports}")
                print("  Specify with --port")
                sys.exit(1)
            else:
                print("  No serial ports found. Connect a USB-UART adapter.")
                sys.exit(1)
        except ImportError:
            print("  Cannot auto-detect port. Specify with --port")
            sys.exit(1)

    print(f"  Opening {args.port} @ {args.baud} baud...")
    ser = serial.Serial(args.port, args.baud, timeout=RX_TIMEOUT_S)
    time.sleep(0.1)  # let port settle
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    print(f"  Port opened successfully.\n")

    try:
        if args.loopback:
            loopback_test(ser)
        elif args.interactive:
            interactive_mode(ser, X_q, y)
        elif args.index is not None:
            # Single beat
            idx = args.index
            if idx < 0 or idx >= len(X_q):
                print(f"  Index {idx} out of range (0-{len(X_q)-1})")
                sys.exit(1)
            expected = y[idx] if y is not None else None
            print(f"  Sending beat #{idx}...")
            predicted = send_beat(ser, X_q[idx])
            if predicted < 0:
                print(f"  Result: TIMEOUT")
            else:
                print(f"  Predicted: {predicted} ({AAMI_LABELS[predicted]})")
                if expected is not None:
                    print(f"  Expected:  {expected} ({AAMI_LABELS[expected]})")
                    print(f"  {'PASS' if predicted == expected else 'FAIL'}")
        else:
            # Batch test
            run_test(ser, X_q, y, num_tests=args.test, start_idx=args.start)
    except KeyboardInterrupt:
        print("\n  Interrupted by user.")
    finally:
        ser.close()
        print("  Port closed.")


if __name__ == "__main__":
    main()
