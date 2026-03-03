// ============================================================
// HDC-AMC Parameters (Default — override via `include or module params)
// Target: Nexys A7-100T (XC7A100TCSG324-1)
// ============================================================

// These are DEFAULT parameters. The auto-generated hdc_params.vh
// from the Python export should be used for synthesis with trained values.

parameter D             = 4096;     // Hypervector dimension
parameter Q             = 16;       // Quantization levels
parameter Q_BITS        = 4;        // Bits for level index (log2(Q))
parameter NUM_CLASSES   = 11;       // Number of modulation classes
parameter WINDOW_SIZE   = 128;      // Samples per classification window

// Architecture parameters
parameter CHUNK_W       = 32;       // Bits processed per clock cycle
parameter NUM_CHUNKS    = 128;      // D / CHUNK_W
parameter INPUT_W       = 8;        // ADC input width (bits) per channel

// Derived parameters
parameter COUNTER_W     = 8;        // Counter width for bundler (log2(WINDOW_SIZE)+1)
parameter DIST_W        = 13;       // Hamming distance width (log2(D)+1)
parameter CLASS_W       = 4;        // Class ID width (ceil(log2(NUM_CLASSES)))
parameter CHUNK_ADDR_W  = 7;        // log2(NUM_CHUNKS)

// Memory depths
parameter CB_DEPTH      = 2048;     // Codebook BRAM depth (Q * NUM_CHUNKS)
parameter CB_ADDR_W     = 11;       // log2(CB_DEPTH)
parameter PROTO_DEPTH   = 1408;     // Prototype BRAM depth (NUM_CLASSES * NUM_CHUNKS)
parameter PROTO_ADDR_W  = 11;       // ceil(log2(PROTO_DEPTH))
