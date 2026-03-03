"""
FPGA Export Module
Exports trained HDC model (codebooks + prototypes) to formats
that can be loaded into Verilog BRAM/ROM via $readmemh or .coe files.

Output Formats:
  - .hex files: Verilog $readmemh compatible
  - .coe files: Vivado Block RAM IP initialization
  - .vh files: Verilog `include headers with parameters
  - Test vectors: I/Q input data + expected classification for RTL testbench

Target: Nexys A7-100T (XC7A100TCSG324-1)
"""

import numpy as np
import os
from typing import List
from .hdc_encoder import HDCEncoder
from .hdc_classifier import HDCClassifier


def bits_to_hex_string(bits: np.ndarray, chunk_width: int = 32) -> List[str]:
    """
    Convert a binary vector to a list of hex strings (MSB first per chunk).
    
    For a D-bit vector, produces D/chunk_width hex strings.
    Each chunk_width bits → one hex value.
    
    Args:
        bits: Binary array of shape (D,) with values 0/1
        chunk_width: Bits per hex word (must be multiple of 4)
        
    Returns:
        List of hex strings (without '0x' prefix), MSB-first within each chunk
    """
    D = len(bits)
    assert D % chunk_width == 0, f"D={D} not divisible by chunk_width={chunk_width}"

    num_chunks = D // chunk_width
    hex_strings = []

    for c in range(num_chunks):
        chunk = bits[c * chunk_width : (c + 1) * chunk_width]
        # Pack bits into integer (MSB first)
        val = 0
        for bit in chunk:
            val = (val << 1) | int(bit)
        hex_str = format(val, f'0{chunk_width // 4}X')
        hex_strings.append(hex_str)

    return hex_strings


def export_codebook_hex(codebook: np.ndarray, filepath: str,
                        chunk_width: int = 32):
    """
    Export a codebook (Q entries × D bits) to Verilog $readmemh format.
    
    File format: One hex word per line, entries ordered as:
      Level 0, Chunk 0
      Level 0, Chunk 1
      ...
      Level 0, Chunk N-1
      Level 1, Chunk 0
      ...
    
    Args:
        codebook: Binary array of shape (Q, D)
        filepath: Output .hex file path
        chunk_width: Bits per word
    """
    Q, D = codebook.shape
    num_chunks = D // chunk_width

    with open(filepath, 'w') as f:
        f.write(f"// HDC Codebook: {Q} levels × {D} bits\n")
        f.write(f"// Chunk width: {chunk_width} bits ({chunk_width//4} hex digits)\n")
        f.write(f"// Total words: {Q * num_chunks}\n")
        f.write(f"// Address layout: level_idx * {num_chunks} + chunk_idx\n")

        for level in range(Q):
            hex_words = bits_to_hex_string(codebook[level], chunk_width)
            for chunk_idx, hw in enumerate(hex_words):
                f.write(f"{hw}\n")

    print(f"[Export] Codebook ({Q}×{D}) → {filepath} "
          f"({Q * num_chunks} words × {chunk_width}b)")


def export_prototypes_hex(prototypes: np.ndarray, filepath: str,
                          chunk_width: int = 32):
    """
    Export class prototype vectors to Verilog $readmemh format.
    
    Args:
        prototypes: Binary array of shape (num_classes, D)
        filepath: Output .hex file path
        chunk_width: Bits per word
    """
    num_classes, D = prototypes.shape
    num_chunks = D // chunk_width

    with open(filepath, 'w') as f:
        f.write(f"// HDC Prototypes: {num_classes} classes × {D} bits\n")
        f.write(f"// Chunk width: {chunk_width} bits ({chunk_width//4} hex digits)\n")
        f.write(f"// Total words: {num_classes * num_chunks}\n")
        f.write(f"// Address layout: class_idx * {num_chunks} + chunk_idx\n")

        for cls in range(num_classes):
            hex_words = bits_to_hex_string(prototypes[cls], chunk_width)
            for chunk_idx, hw in enumerate(hex_words):
                f.write(f"{hw}\n")

    print(f"[Export] Prototypes ({num_classes}×{D}) → {filepath} "
          f"({num_classes * num_chunks} words × {chunk_width}b)")


def export_test_vectors(X_test: np.ndarray, y_test: np.ndarray,
                        y_pred: np.ndarray, encoder: HDCEncoder,
                        output_dir: str, num_vectors: int = 100,
                        input_width: int = 8):
    """
    Export test vectors for RTL simulation verification.
    
    Creates:
      - test_input.hex: I/Q sample pairs as hex
      - test_expected.hex: Expected class labels
      - test_query_vectors.hex: Expected encoded query vectors
    
    Args:
        X_test: Test data shape (N, 2, window_size)
        y_test: True labels
        y_pred: Predicted labels (from Python model)
        encoder: HDCEncoder to generate golden query vectors
        output_dir: Directory for test vector files
        num_vectors: Number of test cases
        input_width: Bit width for I/Q input samples
    """
    os.makedirs(output_dir, exist_ok=True)
    num_vectors = min(num_vectors, len(y_test))

    # Select test vectors where Python prediction matches truth (for clean validation)
    correct_mask = (y_test == y_pred)
    if correct_mask.sum() >= num_vectors:
        indices = np.where(correct_mask)[0][:num_vectors]
    else:
        indices = np.arange(num_vectors)

    window_size = X_test.shape[2]

    # Export input I/Q data
    input_path = os.path.join(output_dir, "test_input.hex")
    with open(input_path, 'w') as f:
        f.write(f"// Test I/Q data: {num_vectors} windows × {window_size} samples\n")
        f.write(f"// Format: one line per sample, upper byte = I, lower byte = Q\n")
        f.write(f"// Window boundaries every {window_size} lines\n")

        for idx in indices:
            I_data = X_test[idx, 0, :]
            Q_data = X_test[idx, 1, :]

            for t in range(window_size):
                # Convert float [-1, 1] to unsigned 8-bit [0, 255]
                I_uint8 = int(np.clip((I_data[t] + 1.0) / 2.0 * 255, 0, 255))
                Q_uint8 = int(np.clip((Q_data[t] + 1.0) / 2.0 * 255, 0, 255))
                f.write(f"{I_uint8:02X}{Q_uint8:02X}\n")

    # Export expected labels
    labels_path = os.path.join(output_dir, "test_expected.hex")
    with open(labels_path, 'w') as f:
        f.write(f"// Expected class labels for {num_vectors} test vectors\n")
        for idx in indices:
            f.write(f"{y_pred[idx]:02X}\n")

    # Export golden query vectors (Python-encoded)
    query_path = os.path.join(output_dir, "test_query_vectors.hex")
    with open(query_path, 'w') as f:
        f.write(f"// Golden query vectors: {num_vectors} × {encoder.D} bits\n")
        f.write(f"// 32-bit hex words, {encoder.D // 32} words per vector\n")

        for idx in indices:
            query = encoder.encode_window_vectorized(
                X_test[idx, 0, :], X_test[idx, 1, :]
            )
            hex_words = bits_to_hex_string(query, chunk_width=32)
            for hw in hex_words:
                f.write(f"{hw}\n")

    print(f"[Export] Test vectors ({num_vectors} cases) → {output_dir}/")
    print(f"  Input: {input_path}")
    print(f"  Labels: {labels_path}")
    print(f"  Queries: {query_path}")


def export_verilog_params(D: int, Q: int, n_gram: int, num_classes: int,
                          window_size: int, chunk_width: int,
                          filepath: str):
    """
    Export Verilog parameter header file.
    Generates an `include file with all HDC parameters for RTL.
    
    Args:
        D, Q, n_gram, num_classes, window_size, chunk_width: Parameters
        filepath: Output .vh file path
    """
    num_chunks = D // chunk_width
    codebook_depth = Q * num_chunks
    prototype_depth = num_classes * num_chunks
    counter_width = (window_size - 1).bit_length() + 1  # log2(window_size) + 1
    dist_width = D.bit_length()  # log2(D) + 1 bits for Hamming distance
    class_width = (num_classes - 1).bit_length()

    with open(filepath, 'w') as f:
        f.write(f"// ============================================================\n")
        f.write(f"// HDC-AMC Parameters (Auto-generated by export_to_fpga.py)\n")
        f.write(f"// Target: Nexys A7-100T (XC7A100TCSG324-1)\n")
        f.write(f"// ============================================================\n\n")

        f.write(f"// Core HDC parameters\n")
        f.write(f"parameter D             = {D};      // Hypervector dimension\n")
        f.write(f"parameter Q             = {Q};       // Quantization levels\n")
        f.write(f"parameter Q_BITS        = {(Q-1).bit_length()};        // Bits for level index (log2(Q))\n")
        f.write(f"parameter N_GRAM        = {n_gram};        // N-gram length\n")
        f.write(f"parameter NUM_CLASSES   = {num_classes};       // Number of modulation classes\n")
        f.write(f"parameter WINDOW_SIZE   = {window_size};     // Samples per classification window\n")
        f.write(f"\n")
        f.write(f"// Architecture parameters\n")
        f.write(f"parameter CHUNK_W       = {chunk_width};       // Bits processed per clock cycle\n")
        f.write(f"parameter NUM_CHUNKS    = {num_chunks};     // D / CHUNK_W\n")
        f.write(f"parameter INPUT_W       = 8;        // ADC input width (bits) per channel\n")
        f.write(f"\n")
        f.write(f"// Derived parameters\n")
        f.write(f"parameter COUNTER_W     = {counter_width};        // Counter width for bundler\n")
        f.write(f"parameter DIST_W        = {dist_width};       // Hamming distance width (log2(D)+1)\n")
        f.write(f"parameter CLASS_W       = {class_width};        // Class ID width\n")
        f.write(f"\n")
        f.write(f"// Memory depths\n")
        f.write(f"parameter CB_DEPTH      = {codebook_depth};     // Codebook BRAM depth (Q * NUM_CHUNKS)\n")
        f.write(f"parameter PROTO_DEPTH   = {prototype_depth};   // Prototype BRAM depth (NUM_CLASSES * NUM_CHUNKS)\n")

    print(f"[Export] Verilog parameters → {filepath}")


def export_coe_file(data: np.ndarray, filepath: str, chunk_width: int = 32):
    """
    Export data as Vivado .coe file for Block RAM IP initialization.
    
    Args:
        data: Binary array of shape (num_entries, D)
        filepath: Output .coe file path
        chunk_width: Bits per word
    """
    num_entries, D = data.shape
    num_chunks = D // chunk_width

    with open(filepath, 'w') as f:
        f.write(f"; HDC Memory Initialization File\n")
        f.write(f"; {num_entries} entries × {D} bits = {num_entries * num_chunks} words × {chunk_width} bits\n")
        f.write(f"memory_initialization_radix=16;\n")
        f.write(f"memory_initialization_vector=\n")

        all_words = []
        for entry in range(num_entries):
            hex_words = bits_to_hex_string(data[entry], chunk_width)
            all_words.extend(hex_words)

        for i, word in enumerate(all_words):
            if i < len(all_words) - 1:
                f.write(f"{word},\n")
            else:
                f.write(f"{word};\n")

    print(f"[Export] COE file → {filepath}")


def export_all(classifier: HDCClassifier, X_test: np.ndarray,
               y_test: np.ndarray, y_pred: np.ndarray,
               output_dir: str, chunk_width: int = 32,
               window_size: int = 128):
    """
    Export everything needed for FPGA implementation.
    
    Creates:
      - codebook_i.hex, codebook_q.hex: Codebook ROM contents
      - prototypes.hex: Class prototype vectors
      - hdc_params.vh: Verilog parameter include file
      - test_vectors/: Test data for RTL simulation
      - *.coe: Vivado Block RAM init files
    """
    os.makedirs(output_dir, exist_ok=True)

    encoder = classifier.encoder
    D = encoder.D
    Q = encoder.Q

    # Export codebooks
    export_codebook_hex(encoder.codebook_I,
                        os.path.join(output_dir, "codebook_i.hex"), chunk_width)
    export_codebook_hex(encoder.codebook_Q,
                        os.path.join(output_dir, "codebook_q.hex"), chunk_width)

    # Export prototypes
    export_prototypes_hex(classifier.prototypes,
                          os.path.join(output_dir, "prototypes.hex"), chunk_width)

    # Export COE files (for Vivado Block RAM IP)
    export_coe_file(encoder.codebook_I,
                    os.path.join(output_dir, "codebook_i.coe"), chunk_width)
    export_coe_file(encoder.codebook_Q,
                    os.path.join(output_dir, "codebook_q.coe"), chunk_width)
    export_coe_file(classifier.prototypes,
                    os.path.join(output_dir, "prototypes.coe"), chunk_width)

    # Export Verilog parameters
    export_verilog_params(
        D=D, Q=Q, n_gram=encoder.n_gram,
        num_classes=classifier.num_classes,
        window_size=window_size,
        chunk_width=chunk_width,
        filepath=os.path.join(output_dir, "hdc_params.vh")
    )

    # Export test vectors (same directory as other hex files)
    export_test_vectors(X_test, y_test, y_pred, encoder, output_dir,
                        num_vectors=50)

    print(f"\n[Export] All FPGA files exported to {output_dir}/")
    print(f"  Copy .hex files to your Vivado project for simulation")
    print(f"  Use .coe files for Block RAM IP initialization")
    print(f"  Include hdc_params.vh in your Verilog source")
