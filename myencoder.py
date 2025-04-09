import numpy as np
from scipy.fftpack import dct
import sys
import os
import time

def rgb_to_yuv(rgb):
    r = rgb[:,:,0]
    g = rgb[:,:,1]
    b = rgb[:,:,2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y

class RGBVideoReader:
    def __init__(self, rgb_path, width=352, height=288):
        if width % 16 != 0 or height % 16 != 0:
            raise ValueError("Video dimensions must be multiple of 16")
        
        self.file = open(rgb_path, 'rb')
        self.width = width
        self.height = height
        self.frame_size = width * height * 3
        
    def read_frame(self):
        raw_data = self.file.read(self.frame_size)
        if len(raw_data) < self.frame_size:
            return False, None
        
        frame = np.frombuffer(raw_data, dtype=np.uint8)
        frame = frame.reshape((self.height, self.width, 3))
        return True, frame
        
    def release(self):
        self.file.close()

class VideoEncoder:
    def __init__(self, input_path, n1, n2, width=352, height=288):
        self.reader = RGBVideoReader(input_path, width, height)
        self.n1 = int(n1)
        self.n2 = int(n2)
        self.width = width
        self.height = height

    def compute_motion_vectors(self, prev_y, curr_y):
        motion_vectors = np.zeros((self.height // 16, self.width // 16, 2))
        block_positions = [(i, j) 
                          for i in range(0, self.height - 16 + 1, 16)
                          for j in range(0, self.width - 16 + 1, 16)]
        
        for i, j in block_positions:
            block = curr_y[i:i+16, j:j+16]
            min_mad = float('inf')
            best_motion = [0, 0]
            step_sizes = [8, 4, 2]
            dx, dy = 0, 0
            
            for step in step_sizes:
                for delta_y in [-step, 0, step]:
                    for delta_x in [-step, 0, step]:
                        new_y = i + dy + delta_y
                        new_x = j + dx + delta_x
                        
                        if (new_y >= 0 and new_y + 16 <= self.height and 
                            new_x >= 0 and new_x + 16 <= self.width):
                            
                            ref_block = prev_y[new_y:new_y+16, new_x:new_x+16]
                            mad = np.mean(np.abs(block - ref_block))
                            
                            if mad < min_mad:
                                min_mad = mad
                                best_motion = [dx + delta_x, dy + delta_y]
                
                dx, dy = best_motion
                
            motion_vectors[i//16, j//16] = best_motion
                
        return motion_vectors

    def classify_macroblocks(self, motion_vectors):
        magnitudes = np.sqrt(np.sum(motion_vectors**2, axis=2))
        threshold = 2.0
        return (magnitudes > threshold).astype(np.uint8)

    def dct_block(self, block):
        return dct(dct(block.T, norm='ortho').T, norm='ortho')

    def compress_frame(self, frame, block_types):
        compressed_blocks = []
        positions = [(i, j, c) 
                    for i in range(0, self.height - 8 + 1, 8)
                    for j in range(0, self.width - 8 + 1, 8)
                    for c in range(3)]
        
        for i, j, c in positions:
            mb_type = block_types[i//16, j//16]
            q_step = 2 ** (self.n1 if mb_type == 1 else self.n2)
            
            block = frame[i:i+8, j:j+8, c]
            dct_coeffs = self.dct_block(block)
            quantized = np.round(dct_coeffs / q_step) * q_step
            
            nonzero_mask = quantized != 0
            if np.any(nonzero_mask):
                compressed_blocks.append({
                    'pos': (i, j),
                    'channel': c,
                    'type': mb_type,
                    'coeffs': quantized[nonzero_mask],
                    'mask': nonzero_mask
                })
                    
        return compressed_blocks

    def encode(self):
        start_time = time.time()
        output_path = 'input_video.cmp'
        ret, prev_frame = self.reader.read_frame()
        if not ret:
            print("Error: Could not read input video")
            return

        all_frames = []
        frame_count = 0
        prev_y = rgb_to_yuv(prev_frame)
        
        print("Encoding video...")
        while True:
            ret, curr_frame = self.reader.read_frame()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {frame_count} frames... ({elapsed:.2f} seconds)")
                
            curr_y = rgb_to_yuv(curr_frame)
            vectors = self.compute_motion_vectors(prev_y, curr_y)
            block_types = self.classify_macroblocks(vectors)
            compressed = self.compress_frame(curr_frame, block_types)
            all_frames.append(compressed)
            
            prev_y = curr_y
        
        elapsed = time.time() - start_time
        print(f"Encoding completed in {elapsed:.2f} seconds")
        
        compressed_data = {
            'n1': self.n1,
            'n2': self.n2,
            'dimensions': (self.width, self.height),
            'frames': all_frames
        }
        
        try:
            with open(output_path, 'wb') as f:
                f.write(np.int32(self.n1).tobytes())
                f.write(np.int32(self.n2).tobytes())
                f.write(np.int32(self.width).tobytes())
                f.write(np.int32(self.height).tobytes())
                f.write(np.int32(len(all_frames)).tobytes())
                
                for frame_data in all_frames:
                    f.write(np.int32(len(frame_data)).tobytes())
                    
                    for block in frame_data:
                        pos = block['pos']
                        f.write(np.int32(pos[0]).tobytes())
                        f.write(np.int32(pos[1]).tobytes())
                        f.write(np.int32(block['channel']).tobytes())
                        f.write(np.int32(block['type']).tobytes())
                        
                        coeffs = block['coeffs']
                        mask = block['mask']
                        f.write(np.int32(len(coeffs)).tobytes())
                        f.write(coeffs.tobytes())
                        f.write(mask.tobytes())
            
            print(f"Compressed file saved to {output_path}")
            file_size = os.path.getsize(output_path)
            print(f"Compressed file size: {file_size} bytes")
            
        except Exception as e:
            print(f"Error saving compressed file: {str(e)}")
        
        self.reader.release()

def main():
    if len(sys.argv) != 4:
        print("Usage: python myencoder.py input_video.rgb n1 n2")
        sys.exit(1)
        
    input_path = sys.argv[1]
    n1 = sys.argv[2]
    n2 = sys.argv[3]
    
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found")
        sys.exit(1)
        
    try:
        encoder = VideoEncoder(input_path, n1, n2)
        encoder.encode()
    except Exception as e:
        print(f"Error during encoding: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
