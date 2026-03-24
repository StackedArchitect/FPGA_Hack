`timescale 1ns / 1ps

module tb_debug;
    localparam CLK_PERIOD = 10;
    localparam NUM_SAMPLES = 187;

    reg         clk = 0;
    reg         rst_n = 0;
    reg         start = 0;
    reg  signed [7:0] sample = 0;
    reg         sample_valid = 0;

    wire        busy, done;
    wire [2:0]  pred_class;

    wavebnn_core u_dut (
        .clk(clk), .rst_n(rst_n),
        .i_start(start), .i_sample(sample), .i_sample_valid(sample_valid),
        .o_busy(busy), .o_done(done), .o_class(pred_class)
    );

    always #(CLK_PERIOD/2) clk = ~clk;

    // Load test vector
    reg [7:0] ecg_mem [0:NUM_SAMPLES-1];
    initial $readmemh("test_input.mem", ecg_mem); // only reads first 187 values

    // Monitor core state
    always @(posedge clk) begin
        if (u_dut.state != u_dut.state) ; // dummy
    end

    // Print state changes
    reg [2:0] prev_state = 0;
    always @(posedge clk) begin
        if (u_dut.state !== prev_state) begin
            $display("T=%0t core_state=%0d wav_state=%0d", $time, u_dut.state,
                     u_dut.u_wavelet.state);
            prev_state <= u_dut.state;
        end
    end

    // Print wavelet done
    always @(posedge clk) begin
        if (u_dut.wav_done) $display("T=%0t WAV_DONE", $time);
        if (u_dut.br_cA3_done) $display("T=%0t BR_CA3_DONE", $time);
        if (u_dut.br_cD3_done) $display("T=%0t BR_CD3_DONE", $time);
        if (u_dut.br_cD2_done) $display("T=%0t BR_CD2_DONE", $time);
        if (u_dut.br_cD1_done) $display("T=%0t BR_CD1_DONE", $time);
        if (u_dut.fc1_done)    $display("T=%0t FC1_DONE", $time);
        if (u_dut.fc2_done)    $display("T=%0t FC2_DONE", $time);
        if (done)              $display("T=%0t PIPELINE_DONE class=%0d", $time, pred_class);
    end

    integer s;
    initial begin
        // Reset
        repeat (10) @(posedge clk);
        rst_n = 1;
        repeat (5) @(posedge clk);

        $display("=== Debug: Single Test ===");

        // Send start + first sample (#1 avoids delta-cycle race)
        @(posedge clk); #1;
        start = 1;
        sample = $signed(ecg_mem[0]);
        sample_valid = 1;

        @(posedge clk); #1;
        start = 0;

        // Feed remaining 186 samples
        for (s = 1; s < NUM_SAMPLES; s = s + 1) begin
            sample = $signed(ecg_mem[s]);
            sample_valid = 1;
            @(posedge clk); #1;
        end
        sample_valid = 0;

        // Feed remaining 186 samples
        for (s = 1; s < NUM_SAMPLES; s = s + 1) begin
            sample = $signed(ecg_mem[s]);
            sample_valid = 1;
            @(posedge clk);
        end
        sample_valid = 0;

        // Wait for done (max 10000 cycles)
        repeat (10000) begin
            @(posedge clk);
            if (done) begin
                $display("SUCCESS: class=%0d", pred_class);
                $finish;
            end
        end

        $display("TIMEOUT: pipeline did not complete in 10000 cycles");
        $display("  core_state=%0d", u_dut.state);
        $display("  wav_start=%0b i_start=%0b", u_dut.wav_start, u_dut.i_start);
        $display("  wav_state=%0d wav_busy=%0b wav_done=%0b",
                 u_dut.u_wavelet.state, u_dut.wav_busy, u_dut.wav_done);
        $display("  wav_sample_cnt=%0d", u_dut.u_wavelet.sample_cnt);
        $display("  br_cA3: state=%0d done=%0b finished=%0b",
                 u_dut.u_br_cA3.state, u_dut.br_cA3_done, u_dut.br_cA3_finished);
        $display("  br_cD1: state=%0d done=%0b finished=%0b load_cnt=%0d pos=%0d",
                 u_dut.u_br_cD1.state, u_dut.br_cD1_done, u_dut.br_cD1_finished,
                 u_dut.u_br_cD1.load_cnt, u_dut.u_br_cD1.pos);
        $display("  fc1: state=%0d neuron=%0d", u_dut.u_fc1.state, u_dut.u_fc1.neuron_idx);
        $display("  fc2: state=%0d bit=%0d", u_dut.u_fc2.state, u_dut.u_fc2.bit_idx);
        $display("  feed_addr=%0d feed_active=%0b", u_dut.feed_addr, u_dut.feed_active);
        $finish;
    end

    // Hard watchdog
    initial begin
        #(CLK_PERIOD * 50000);
        $display("HARD WATCHDOG at T=%0t", $time);
        $display("  core_state=%0d wav_state=%0d", u_dut.state, u_dut.u_wavelet.state);
        $finish;
    end
endmodule
