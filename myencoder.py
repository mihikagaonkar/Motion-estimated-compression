import numpy as np
from scipy.fftpack import dct
import sys
import os
import time
from threading import Thread
from queue import Queue

def rgb_to_yuv(rgb):
    # Converts an RGB frame to a single-channel Y (luminance) component in YUV color space
    r = rgb[:,:,0]
    g = rgb[:,:,1]
    b = rgb[:,:,2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y

class RGBVideoReader:
    def __init__(self, rgb_path, width=960, height=540):
        # Initializes the video reader for a raw RGB video file
        self.file = open(rgb_path, 'rb')
        self.width = width
        self.height = height
        self.frame_size = width * height * 3  # RGB = 3 channels
        
    def read_frame(self):
        # Reads a single frame from the RGB file
        raw_data = self.file.read(self.frame_size)
        if len(raw_data) < self.frame_size:
            return False, None
            
        frame = np.frombuffer(raw_data, dtype=np.uint8)
        frame = frame.reshape((self.height, self.width, 3))
        return True, frame
        
    def release(self):
        # Closes the video file
        self.file.close()

class VideoEncoder:
    def __init__(self, input_path, n1, n2, width=960, height=540):
        # Initializes the encoder with input video path and quantization parameters
        self.reader = RGBVideoReader(input_path, width, height)
        self.n1 = int(n1)  # Quantization level for moving macroblocks
        self.n2 = int(n2)  # Quantization level for static macroblocks
        self.width = width
        self.height = height
        self.input_path = input_path
        self.frame_queue = Queue(maxsize=50)  # Queue for storing frames to be saved

    def compute_motion_vectors(self, prev_y, curr_y):
        # Computes motion vectors between two luminance frames using a hierarchical search
        rows = (self.height + 15) // 16
        cols = (self.width + 15) // 16
        motion_vectors = np.zeros((rows, cols, 2))  # Stores motion vectors for each macroblock
        
        block_positions = [(i, j) 
                          for i in range(0, self.height, 16)
                          for j in range(0, self.width, 16)]
        
        for i, j in block_positions:
            # Define current macroblock dimensions
            block_h = min(16, self.height - i)
            block_w = min(16, self.width - j)
            block = curr_y[i:i+block_h, j:j+block_w]
            
            min_mad = float('inf')  # Minimum Mean Absolute Difference
            best_motion = [0, 0]
            
            # Hierarchical search steps
            step_sizes = [8, 4, 2]
            dx, dy = 0, 0
            
            for step in step_sizes:
                for delta_y in [-step, 0, step]:
                    for delta_x in [-step, 0, step]:
                        new_y = i + dy + delta_y
                        new_x = j + dx + delta_x
                        
                        if (new_y >= 0 and new_y + block_h <= self.height and 
                            new_x >= 0 and new_x + block_w <= self.width):
                            
                            ref_block = prev_y[new_y:new_y+block_h, new_x:new_x+block_w]
                            mad = np.mean(np.abs(block - ref_block))  # Compute MAD
                            
                            if mad < min_mad:
                                min_mad = mad
                                best_motion = [dx + delta_x, dy + delta_y]
                
                dx, dy = best_motion
            
            motion_vectors[i//16, j//16] = best_motion  # Store best vector
                
        return motion_vectors

    def classify_macroblocks(self, motion_vectors):
        # Classifies macroblocks as moving (1) or static (0) based on vector magnitude
        magnitudes = np.sqrt(np.sum(motion_vectors**2, axis=2))
        threshold = 2.0
        return (magnitudes > threshold).astype(np.uint8)

    def dct_block(self, block):
        # Applies 2D Discrete Cosine Transform to an 8x8 block
        return dct(dct(block.T, norm='ortho').T, norm='ortho')

    def compress_frame(self, frame, block_types):
        # Compresses a single RGB frame using DCT and quantization
        compressed_blocks = []
        
        # Interleaved processing: 8x8 blocks per channel
        positions = [(i, j, c) 
                    for i in range(0, self.height, 8)
                    for j in range(0, self.width, 8)
                    for c in range(3)]
        
        for i, j, c in positions:
            block_h = min(8, self.height - i)
            block_w = min(8, self.width - j)
            
            # Handle edge padding
            if block_h < 8 or block_w < 8:
                block = np.zeros((8, 8), dtype=frame.dtype)
                block[:block_h, :block_w] = frame[i:i+block_h, j:j+block_w, c]
                if block_w < 8:
                    block[:, block_w:] = block[:, block_w-1:block_w]
                if block_h < 8:
                    block[block_h:, :] = block[block_h-1:block_h, :]
            else:
                block = frame[i:i+8, j:j+8, c]
            
            # Choose quantization step based on macroblock type
            mb_type = block_types[i//16, j//16]
            q_step = 2 ** (self.n1 if mb_type == 1 else self.n2)
            
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

    def save_worker(self, output_path):
        # Background thread to save compressed frame data to disk
        try:
            with open(output_path, 'wb') as f:
                header_size = 5 * 4  # Reserve space for header
                f.write(bytes(header_size))
                
                frame_count = 0
                
                while True:
                    frame_data = self.frame_queue.get()
                    if frame_data is None:
                        break
                    
                    # Write frame size
                    f.write(np.int32(len(frame_data)).tobytes())
                    
                    # Write block data for each frame
                    for block in frame_data:
                        metadata = np.array([block['pos'][0], block['pos'][1],
                                          block['channel'], block['type'],
                                          len(block['coeffs'])], dtype=np.int32)
                        f.write(metadata.tobytes())
                        f.write(block['coeffs'].tobytes())
                        f.write(block['mask'].tobytes())
                    
                    frame_count += 1
                
                # Go back and write the header now that we know the total frame count
                f.seek(0)
                header = np.array([self.n1, self.n2, self.width, self.height, 
                                 frame_count], dtype=np.int32)
                f.write(header.tobytes())
                
        except Exception as e:
            print(f"Error in save worker: {str(e)}")

    def encode(self):
        # Main function to read, process, compress, and save video frames
        start_time = time.time()
        input_name = os.path.splitext(os.path.basename(self.input_path))[0]
        output_path = f"{input_name}.cmp"
        
        # Start save thread
        save_thread = Thread(target=self.save_worker, args=(output_path,))
        save_thread.start()
        
        # Read first frame
        ret, prev_frame = self.reader.read_frame()
        if not ret:
            print("Error: Could not read input video")
            return

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
            
            self.frame_queue.put(compressed)
            prev_y = curr_y
        
        # Signal save thread to finish
        self.frame_queue.put(None)
        save_thread.join()
        
        elapsed = time.time() - start_time
        print(f"Encoding completed in {elapsed:.2f} seconds")
        print(f"Compressed file saved to {output_path}")
        print(f"Compressed file size: {os.path.getsize(output_path)} bytes")
        
        self.reader.release()

def main():
    # CLI entry point: expects input file and two quantization parameters
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