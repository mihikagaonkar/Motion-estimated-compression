import numpy as np
import cv2
import pygame
import sys
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

class VideoDecoder:
    def __init__(self, compressed_path, audio_path):
        with open(compressed_path, 'rb') as f:
            self.n1 = np.frombuffer(f.read(4), dtype=np.int32)[0]
            self.n2 = np.frombuffer(f.read(4), dtype=np.int32)[0]
            self.width = np.frombuffer(f.read(4), dtype=np.int32)[0]
            self.height = np.frombuffer(f.read(4), dtype=np.int32)[0]
            self.num_frames = np.frombuffer(f.read(4), dtype=np.int32)[0]

            self.frames = []
            for _ in range(self.num_frames):
                num_blocks = np.frombuffer(f.read(4), dtype=np.int32)[0]
                frame_data = []
                for _ in range(num_blocks):
                    pos_i = np.frombuffer(f.read(4), dtype=np.int32)[0]
                    pos_j = np.frombuffer(f.read(4), dtype=np.int32)[0]
                    channel = np.frombuffer(f.read(4), dtype=np.int32)[0]
                    block_type = np.frombuffer(f.read(4), dtype=np.int32)[0]
                    num_coeffs = np.frombuffer(f.read(4), dtype=np.int32)[0]
                    coeffs = np.frombuffer(f.read(num_coeffs * 8), dtype=np.float64)
                    mask = np.frombuffer(f.read(64), dtype=np.bool_).reshape((8, 8))
                    frame_data.append({
                        'pos': (pos_i, pos_j),
                        'channel': channel,
                        'type': block_type,
                        'coeffs': coeffs,
                        'mask': mask
                    })
                self.frames.append(frame_data)

        self.audio_path = audio_path
        self.frame_rate = 30
        self.current_frame = 0
        self.playing = True
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.frame_cache = OrderedDict()
        self.prefetched_frames = {}

    def idct_block(self, block):
        return cv2.idct(block.astype(np.float32)).astype(np.float32)

    def decompress_frame(self, frame_data):
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for block in frame_data:
            i, j = block['pos']
            c = block['channel']
            dct_block = np.zeros((8, 8), dtype=np.float32)
            dct_block[block['mask']] = block['coeffs']
            pixel_block = self.idct_block(dct_block)
            pixel_block = np.clip(pixel_block, 0, 255)
            block_h = min(8, self.height - i)
            block_w = min(8, self.width - j)
            frame[i:i + block_h, j:j + block_w, c] = pixel_block[:block_h, :block_w]
        return frame

    def get_frame_async(self, frame_idx):
        if frame_idx in self.prefetched_frames:
            return self.prefetched_frames.pop(frame_idx)
        frame = self.decompress_frame(self.frames[frame_idx])
        return frame

    def prefetch_next_frame(self):
        next_frame_idx = self.current_frame + 1
        if next_frame_idx < self.num_frames and next_frame_idx not in self.prefetched_frames:
            future = self.executor.submit(self.decompress_frame, self.frames[next_frame_idx])
            self.prefetched_frames[next_frame_idx] = future.result()

    def play(self):
        pygame.init()
        pygame.display.set_caption('Video Player')
        screen = pygame.display.set_mode((self.width, self.height))
        clock = pygame.time.Clock()
        pygame.mixer.init(frequency=44100)
        if os.path.exists(self.audio_path):
            pygame.mixer.music.load(self.audio_path)
            pygame.mixer.music.play()

        print("\nPlayer Controls:")
        print("Space: Play/Pause")
        print("Right Arrow: Step forward (when paused)")
        print("Left Arrow: Step backward (when paused)")
        print("Q: Quit")

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.playing = not self.playing
                        if self.playing:
                            pygame.mixer.music.unpause()
                        else:
                            pygame.mixer.music.pause()
                    elif event.key == pygame.K_RIGHT and not self.playing:
                        self.current_frame = min(self.current_frame + 1, self.num_frames - 1)
                    elif event.key == pygame.K_LEFT and not self.playing:
                        self.current_frame = max(self.current_frame - 1, 0)

            if self.playing and self.current_frame < self.num_frames:
                frame = self.get_frame_async(self.current_frame)
                self.prefetch_next_frame()
                surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                screen.blit(surface, (0, 0))
                pygame.display.flip()
                self.current_frame += 1
                clock.tick(self.frame_rate)

def main():
    if len(sys.argv) != 3:
        print("Usage: python mydecoder.py input_video.cmp input_audio.wav")
        sys.exit(1)

    compressed_path = sys.argv[1]
    audio_path = sys.argv[2]

    if not os.path.exists(compressed_path):
        printimport numpy as np
import cv2
import pygame
import sys
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

class VideoDecoder:
    def __init__(self, compressed_path, audio_path):
        with open(compressed_path, 'rb') as f:
            self.n1 = np.frombuffer(f.read(4), dtype=np.int32)[0]
            self.n2 = np.frombuffer(f.read(4), dtype=np.int32)[0]
            self.width = np.frombuffer(f.read(4), dtype=np.int32)[0]
            self.height = np.frombuffer(f.read(4), dtype=np.int32)[0]
            self.num_frames = np.frombuffer(f.read(4), dtype=np.int32)[0]

            self.frames = []
            for _ in range(self.num_frames):
                num_blocks = np.frombuffer(f.read(4), dtype=np.int32)[0]
                frame_data = []
                for _ in range(num_blocks):
                    pos_i = np.frombuffer(f.read(4), dtype=np.int32)[0]
                    pos_j = np.frombuffer(f.read(4), dtype=np.int32)[0]
                    channel = np.frombuffer(f.read(4), dtype=np.int32)[0]
                    block_type = np.frombuffer(f.read(4), dtype=np.int32)[0]
                    num_coeffs = np.frombuffer(f.read(4), dtype=np.int32)[0]
                    coeffs = np.frombuffer(f.read(num_coeffs * 8), dtype=np.float64)
                    mask = np.frombuffer(f.read(64), dtype=np.bool_).reshape((8, 8))
                    frame_data.append({
                        'pos': (pos_i, pos_j),
                        'channel': channel,
                        'type': block_type,
                        'coeffs': coeffs,
                        'mask': mask
                    })
                self.frames.append(frame_data)

        self.audio_path = audio_path
        self.frame_rate = 30
        self.current_frame = 0
        self.playing = True
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.frame_cache = OrderedDict()
        self.prefetched_frames = {}

    def idct_block(self, block):
        return cv2.idct(block.astype(np.float32)).astype(np.float32)

    def decompress_frame(self, frame_data):
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for block in frame_data:
            i, j = block['pos']
            c = block['channel']
            dct_block = np.zeros((8, 8), dtype=np.float32)
            dct_block[block['mask']] = block['coeffs']
            pixel_block = self.idct_block(dct_block)
            pixel_block = np.clip(pixel_block, 0, 255)
            block_h = min(8, self.height - i)
            block_w = min(8, self.width - j)
            frame[i:i + block_h, j:j + block_w, c] = pixel_block[:block_h, :block_w]
        return frame

    def get_frame_async(self, frame_idx):
        if frame_idx in self.prefetched_frames:
            return self.prefetched_frames.pop(frame_idx)
        frame = self.decompress_frame(self.frames[frame_idx])
        return frame

    def prefetch_next_frame(self):
        next_frame_idx = self.current_frame + 1
        if next_frame_idx < self.num_frames and next_frame_idx not in self.prefetched_frames:
            future = self.executor.submit(self.decompress_frame, self.frames[next_frame_idx])
            self.prefetched_frames[next_frame_idx] = future.result()

    def play(self):
        pygame.init()
        pygame.display.set_caption('Video Player')
        screen = pygame.display.set_mode((self.width, self.height))
        clock = pygame.time.Clock()
        pygame.mixer.init(frequency=44100)
        if os.path.exists(self.audio_path):
            pygame.mixer.music.load(self.audio_path)
            pygame.mixer.music.play()

        print("\nPlayer Controls:")
        print("Space: Play/Pause")
        print("Right Arrow: Step forward (when paused)")
        print("Left Arrow: Step backward (when paused)")
        print("Q: Quit")

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.playing = not self.playing
                        if self.playing:
                            pygame.mixer.music.unpause()
                        else:
                            pygame.mixer.music.pause()
                    elif event.key == pygame.K_RIGHT and not self.playing:
                        self.current_frame = min(self.current_frame + 1, self.num_frames - 1)
                    elif event.key == pygame.K_LEFT and not self.playing:
                        self.current_frame = max(self.current_frame - 1, 0)

            if self.playing and self.current_frame < self.num_frames:
                frame = self.get_frame_async(self.current_frame)
                self.prefetch_next_frame()
                surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                screen.blit(surface, (0, 0))
                pygame.display.flip()
                self.current_frame += 1
                clock.tick(self.frame_rate)

def main():
    if len(sys.argv) != 3:
        print("Usage: python mydecoder.py input_video.cmp input_audio.wav")
        sys.exit(1)

    compressed_path = sys.argv[1]
    audio_path = sys.argv[2]

    if not os.path.exists(compressed_path):
        print(f"Error: Compressed video file {compressed_path} not found")
        sys.exit(1)

    decoder = VideoDecoder(compressed_path, audio_path)
    decoder.play()

if __name__ == "__main__":
    main()
(f"Error: Compressed video file {compressed_path} not found")
        sys.exit(1)

    decoder = VideoDecoder(compressed_path, audio_path)
    decoder.play()

if __name__ == "__main__":
    main()