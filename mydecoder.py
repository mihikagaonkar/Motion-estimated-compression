import numpy as np
from scipy.fftpack import idct
import pygame
import sys
import os
import wave
import time

class VideoDecoder:
    def __init__(self, compressed_path, audio_path):
        with open(compressed_path, 'rb') as f:
            self.n1 = np.frombuffer(f.read(4), dtype=np.int32)[0]
            self.n2 = np.frombuffer(f.read(4), dtype=np.int32)[0]
            self.width = np.frombuffer(f.read(4), dtype=np.int32)[0]
            self.height = np.frombuffer(f.read(4), dtype=np.int32)[0]
            num_frames = np.frombuffer(f.read(4), dtype=np.int32)[0]
            
            print(f"Video info: {self.width}x{self.height}, {num_frames} frames")
            print(f"Quantization parameters: n1={self.n1}, n2={self.n2}")
            print("Loading compressed frames...")
            self.frames = []
            for frame_idx in range(num_frames):
                num_blocks = np.frombuffer(f.read(4), dtype=np.int32)[0]
                frame_data = []
                for _ in range(num_blocks):
                    pos_i = np.frombuffer(f.read(4), dtype=np.int32)[0]
                    pos_j = np.frombuffer(f.read(4), dtype=np.int32)[0]
                    channel = np.frombuffer(f.read(4), dtype=np.int32)[0]
                    block_type = np.frombuffer(f.read(4), dtype=np.int32)[0]
                    num_coeffs = np.frombuffer(f.read(4), dtype=np.int32)[0]
                    coeffs = np.frombuffer(f.read(num_coeffs * 8), dtype=np.float64)
                    mask = np.frombuffer(f.read(64), dtype=np.bool_)
                    frame_data.append({
                        'pos': (pos_i, pos_j),
                        'channel': channel,
                        'type': block_type,
                        'coeffs': coeffs,
                        'mask': mask.reshape((8, 8))
                    })
                self.frames.append(frame_data)
                if frame_idx % 10 == 0:
                    print(f"Loaded {frame_idx}/{num_frames} frames...")

        self.audio_path = audio_path
        self.frame_rate = 30
        self.current_frame = 0
        self.playing = False

        print("Pre-processing frames...")
        self.decompressed_frames = []
        for i, frame_data in enumerate(self.frames):
            self.decompressed_frames.append(self.decompress_frame(frame_data))
            if i % 10 == 0:
                print(f"Processed {i}/{len(self.frames)} frames...")
        print("Pre-processing complete")

    def idct_block(self, block):
        return idct(idct(block.T, norm='ortho').T, norm='ortho')
        
    def decompress_frame(self, frame_data):
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for block in frame_data:
            i, j = block['pos']
            c = block['channel']
            dct_block = np.zeros((8, 8))
            dct_block[block['mask']] = block['coeffs']
            pixel_block = self.idct_block(dct_block)
            block_h = min(8, self.height - i)
            block_w = min(8, self.width - j)
            pixel_block = np.clip(pixel_block, 0, 255).astype(np.uint8)
            frame[i:i+block_h, j:j+block_w, c] = pixel_block[:block_h, :block_w]
        return frame

    def play(self):
        pygame.init()
        pygame.display.set_caption('Video Player')
        screen = pygame.display.set_mode((self.width, self.height))
        clock = pygame.time.Clock()
        if os.path.exists(self.audio_path):
            pygame.mixer.init(frequency=44100)
            pygame.mixer.music.load(self.audio_path)
            audio_started = False
        else:
            print(f"Warning: Audio file {self.audio_path} not found")
            audio_started = True

        self.playing = True
        print("\nPlayer Controls:")
        print("Space: Play/Pause")
        print("Right Arrow: Step forward (when paused)")
        print("Left Arrow: Step backward (when paused)")
        print("Q: Quit")

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.playing = not self.playing
                        if self.playing:
                            if not audio_started:
                                pygame.mixer.music.play()
                                audio_started = True
                            else:
                                pygame.mixer.music.unpause()
                        else:
                            pygame.mixer.music.pause()
                    elif event.key == pygame.K_RIGHT and not self.playing:
                        self.current_frame = min(self.current_frame + 1,
                                              len(self.decompressed_frames) - 1)
                    elif event.key == pygame.K_LEFT and not self.playing:
                        self.current_frame = max(self.current_frame - 1, 0)

            if self.current_frame >= len(self.decompressed_frames):
                self.current_frame = 0
                if self.playing:
                    pygame.mixer.music.play()

            frame = self.decompressed_frames[self.current_frame]
            surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            screen.blit(surface, (0, 0))
            pygame.display.flip()

            if self.playing:
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
        
    try:
        decoder = VideoDecoder(compressed_path, audio_path)
        decoder.play()
    except Exception as e:
        print(f"Error during decoding: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
