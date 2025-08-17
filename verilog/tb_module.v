module tb_access_system;

    reg clk, rst;
    reg [3:0] user_id;
    wire [3:0] resource_id;

    // Instantiate DUT
    access_system dut (.clk(clk),
                        .rst(rst),
                        .user_id(user_id),
                        .resource_id(resource_id));

    integer file;
    integer timestamp;

    // Clock generation (10ns period → 100 MHz)
    always #5 clk = ~clk;

    initial 
        begin
            // CSV file open for writing
            file = $fopen("verilog_output.csv", "w");
            $fdisplay(file, "timestamp,user_id,resource_id"); // CSV header

            // Init
            clk = 0; rst = 1; user_id = 0; timestamp = 0;

            #12 rst = 0;  // de-assert reset

            // Run 50 cycles
            repeat (50) 
                begin
                    @(posedge clk);
                    user_id = $urandom % 10;   // random IDs
                    timestamp = timestamp + 1;
                    $fdisplay(file, "%0d,%0d,%0d", timestamp, user_id, resource_id);
                end

            $fclose(file);
            $finish;
        end
endmodule