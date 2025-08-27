module access_system (input wire clk,
                    input wire rst,
                    input wire [3:0] user_id,     
                    output reg [3:0] resource_id);

    always @(posedge clk or posedge rst) begin
        if (rst) 
            begin
                resource_id <= 0;
            end 
        else 
            begin
                // Simple logic: resource = (user * 3) mod 10
                resource_id <= (user_id * 3) % 10;
            end
    end
endmodule