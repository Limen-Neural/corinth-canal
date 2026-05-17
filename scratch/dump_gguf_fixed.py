import struct
import sys

def read_gguf(path):
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'GGUF':
            return
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]
        
        for _ in range(n_kv):
            key_len = struct.unpack('<Q', f.read(8))[0]
            key = f.read(key_len).decode('utf-8')
            vtype = struct.unpack('<I', f.read(4))[0]
            
            if vtype == 4: val = struct.unpack('<I', f.read(4))[0]
            elif vtype == 5: val = struct.unpack('<i', f.read(4))[0]
            elif vtype == 6: val = struct.unpack('<f', f.read(4))[0]
            elif vtype == 7: val = struct.unpack('<?', f.read(1))[0]
            elif vtype == 8:
                v_len = struct.unpack('<Q', f.read(8))[0]
                val = f.read(v_len).decode('utf-8')
            elif vtype == 9:
                atype = struct.unpack('<I', f.read(4))[0]
                alen = struct.unpack('<Q', f.read(8))[0]
                val = f"ARRAY of {atype} len {alen}"
                # Skip array data
                if atype == 4: f.seek(4 * alen, 1)
                elif atype == 5: f.seek(4 * alen, 1)
                elif atype == 6: f.seek(4 * alen, 1)
                elif atype == 7: f.seek(1 * alen, 1)
                elif atype == 8:
                    for _ in range(alen):
                        slen = struct.unpack('<Q', f.read(8))[0]
                        f.seek(slen, 1)
                elif atype == 11: f.seek(8 * alen, 1)
            elif vtype == 11: val = struct.unpack('<Q', f.read(8))[0]
            else: val = f"Type {vtype}"
            
            if "gemma4" in key or "block" in key or "embd" in key or "head" in key or "attention" in key:
                print(f"{key}: {val}")

if __name__ == "__main__":
    read_gguf(sys.argv[1])
