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
    def __init__(self, rgb_path, width=960, height=540):
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
    def __init__(self, input_path, n1, n2, width=960, height=540):
        self.reader = RGBVideoReader(input_path, width, height)
        self.n1 = int(n1)
        self.n2 = int(n2)
        self.width = width
        self.height = height

    def compute_motion_vectors(self, prev_y, curr_y):
        rows = (self.height + 15) // 16
        cols = (self.width + 15) // 16
        motion_vectors = np.zeros((rows, cols, 2))
        block_positions = [(i, j) for i in range(0, self.height, 16) for j in range(0, self.width, 16)]
        for i, j in block_positions:
            block_h = min(16, self.height - i)
            block_w = min(16, self.width - j)
            block = curr_y[i:i+block_h, j:j+block_w]
            min_mad = float('inf')
            best_motion = [0, 0]
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
                            mad = np.mean(np.abs(block - ref_block))
