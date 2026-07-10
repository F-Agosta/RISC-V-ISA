import re
import struct
import sys

def main():
    input_file = '/Users/francescoagosta/ACA2026/RISC-V-ISA/dataset/model_weights.h'
    output_file = '/Users/francescoagosta/ACA2026/RISC-V-ISA/dataset/model_weights_binary.h'
    
    with open(input_file, 'r') as f:
        content = f.read()
        
    # Find SVM_W
    match = re.search(r'const float SVM_W\[\d+\][^{]*\{\s*([^}]+)\s*\}', content)
    if not match:
        print("Could not find SVM_W array in model_weights.h")
        sys.exit(1)
        
    float_strings = match.group(1).replace('f', '').replace(',', ' ').split()
    floats = [float(s) for s in float_strings]
    
    print(f"Extracted {len(floats)} floats for SVM_W")
    
    if len(floats) != 12288: # 6 classes * 2048 dimensions
        print("Warning: Expected 12288 floats")
        
    binary_uint32 = []
    
    # Process 32 floats at a time
    for i in range(0, len(floats), 32):
        chunk = floats[i:i+32]
        packed_val = 0
        for bit_index, val in enumerate(chunk):
            bit = 1 if val > 0 else 0
            packed_val |= (bit << bit_index)
        binary_uint32.append(packed_val)
        
    print(f"Generated {len(binary_uint32)} uint32_t values")
    
    with open(output_file, 'w') as f:
        f.write("#ifndef MODEL_WEIGHTS_BINARY_H\n")
        f.write("#define MODEL_WEIGHTS_BINARY_H\n\n")
        f.write("#include \"model_weights.h\"\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"const uint32_t SVM_W_BINARY[{len(binary_uint32)}] __attribute__((aligned(64))) = {{\n")
        
        for i, val in enumerate(binary_uint32):
            if i % 8 == 0:
                f.write("    ")
            f.write(f"0x{val:08x}, ")
            if i % 8 == 7:
                f.write("\n")
                
        f.write("};\n\n")
        f.write("#endif // MODEL_WEIGHTS_BINARY_H\n")
        
    print(f"Successfully wrote {output_file}")

if __name__ == "__main__":
    main()
