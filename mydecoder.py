import numpy as np
import cv2
import pygame
import sys
import os
import time
from multiprocessing import Process, Queue, cpu_count
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

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
                    frame_data.append((pos_i, pos_j, channel, coeffs, mask))
                self.frames.append(frame_data)

        self.audio_path = audio_path
        self.frame_rate = 30
        self.playing = False
        self.frame_time = 1.0 / self.frame_rate

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Loading...')

        self.decoded_frames = []
        self._preprocess_frames()

    def _decode_frame(self, frame_data):
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        channel_blocks = [[] for _ in range(3)]
        for pos_i, pos_j, channel, coeffs, mask in frame_data:
            channel_blocks[channel].append((pos_i, pos_j, coeffs, mask))

        for channel, blocks in enumerate(channel_blocks):
            for pos_i, pos_j, coeffs, mask in blocks:
                dct_block = np.zeros((8, 8), dtype=np.float32)
                dct_block[mask] = coeffs.astype(np.float32)
                pixel_block = cv2.idct(dct_block)
                pixel_block = np.clip(pixel_block, 0, 255)
                h, w = min(8, self.height - pos_i), min(8, self.width - pos_j)
                frame[pos_i:pos_i + h, pos_j:pos_j + w, channel] = pixel_block[:h, :w]

        return frame

    def _preprocess_frames(self):
        font = pygame.font.Font(None, 36)

        for i, frame_data in enumerate(self.frames):
            progress = (i + 1) / self.num_frames * 100
            self.screen.fill((0, 0, 0))
            text = font.render(f'Loading: {progress:.1f}%', True, (255, 255, 255))
            text_rect = text.get_rect(center=(self.width/2, self.height/2))
            self.screen.blit(text, text_rect)
            pygame.display.flip()

            frame = self._decode_frame(frame_data)
            self.decoded_frames.append(frame)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

    def _display_frame(self, frame_idx):
        frame = self.decoded_frames[frame_idx]
        surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        self.screen.blit(surface, (0, 0))
        font = pygame.font.Font(None, 36)
        text = font.render(f'Frame: {frame_idx + 1}/{self.num_frames}', True, (255, 255, 255))
        text_rect = text.get_rect(topleft=(10, 10))
        self.screen.blit(text, text_rect)
        pygame.display.flip()

    def play(self):
        clock = pygame.time.Clock()
        current_frame = 0
        key_hold_time = 0
        seek_speed = 1

        if os.path.exists(self.audio_path):
            pygame.mixer.init(frequency=44100)
            pygame.mixer.music.load(self.audio_path)

        print("\nControls:")
        print("Space: Play/Pause")
        print("Left/Right Arrows: Previous/Next Frame (Hold to seek faster)")
        print("Q: Quit")

        self.playing = True
        pygame.mixer.music.play()
        start_time = time.time()

        while current_frame < self.num_frames:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.playing = not self.playing
                    if self.playing:
                        pygame.mixer.music.unpause()
                        start_time = time.time() - (current_frame * self.frame_time)
                    else:
                        pygame.mixer.music.pause()

            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] or keys[pygame.K_RIGHT]:
                key_hold_time += 1
                seek_speed = min(30, 1 + key_hold_time // 30)
            else:
                key_hold_time = 0
                seek_speed = 1

            if keys[pygame.K_LEFT]:
                current_frame = max(0, current_frame - seek_speed)
                self._display_frame(current_frame)
                start_time = time.time() - (current_frame * self.frame_time)
                if self.playing:
                    pygame.mixer.music.play(start=current_frame / self.frame_rate)
                clock.tick(30)
            elif keys[pygame.K_RIGHT]:
                current_frame = min(self.num_frames - 1, current_frame + seek_speed)
                self._display_frame(current_frame)
                start_time = time.time() - (current_frame * self.frame_time)
                if self.playing:
                    pygame.mixer.music.play(start=current_frame / self.frame_rate)
                clock.tick(30)
            elif self.playing:
                current_time = time.time()
                elapsed = current_time - start_time
                target_frame = int(elapsed * self.frame_rate)

                if target_frame < self.num_frames:
                    self._display_frame(target_frame)
                    current_frame = target_frame + 1
                else:
                    current_frame = self.num_frames

            clock.tick(60)

def main():
    if len(sys.argv) != 3:
        print("Usage: python decoder.py input_video.cmp input_audio.wav")
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