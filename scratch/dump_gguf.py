import struct
import sys

def read_gguf(path):
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'GGUF':
            print("Not a GGUF file")
            return
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]
        
        print(f"Version: {version}")
        print(f"Tensors: {n_tensors}")
        print(f"KV pairs: {n_kv}")
        
        for _ in range(n_kv):
            key_len = struct.unpack('<Q', f.read(8))[0]
            key = f.read(key_len).decode('utf-8')
            vtype = struct.unpack('<I', f.read(4))[0]
            
            # Simplified reading for common types
            if vtype == 4: # UINT32
                val = struct.unpack('<I', f.read(4))[0]
            elif vtype == 5: # INT32
                val = struct.unpack('<i', f.read(4))[0]
            elif vtype == 6: # FLOAT32
                val = struct.unpack('<f', f.read(4))[0]
            elif vtype == 7: # BOOL
                val = struct.unpack('<?', f.read(1))[0]
            elif vtype == 8: # STRING
                v_len = struct.unpack('<Q', f.read(8))[0]
                val = f.read(v_len).decode('utf-8')
            else:
                # Skip other types
                val = f"Type {vtype}"
                if vtype == 9: # ARRAY
                    # Very basic skip
                    atype = struct.unpack('<I', f.read(4))[0]
                    alen = struct.unpack('<Q', f.read(8))[0]
                    # This is hard to skip without full parser, but we only care about first few keys
                    print(f"{key}: ARRAY of {atype} len {alen}")
                    break
                elif vtype == 11: # UINT64
                    val = struct.unpack('<Q', f.read(8))[0]
            
            if "gemma4" in key or "block_count" in key or "embedding_length" in key or "head_count" in key:
                print(f"{key}: {val}")

if __name__ == "__main__":
    read_gguf(sys.argv[1])
