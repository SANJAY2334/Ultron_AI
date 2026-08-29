/* ULTRON Native Icon Factory (Phase 4G.4: Stage 4)
 *
 * Generates an in-memory 16x16 RGBA cyan robotic reticle icon for the System Tray.
 * Ensures standalone execution without missing asset file paths.
 */

import { nativeImage, NativeImage } from 'electron';

export function createTrayIcon(): NativeImage {
  // 16x16 RGBA buffer (4 bytes per pixel = 1024 bytes)
  const size = 16;
  const buffer = Buffer.alloc(size * size * 4);

  // Draw a cyberpunk cyan diamond reticle on a transparent background
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const idx = (y * size + x) * 4;
      const isBorder =
        (x === 0 || x === size - 1 || y === 0 || y === size - 1) &&
        (x >= 4 && x <= 11 && (y === 0 || y === size - 1) ||
         y >= 4 && y <= 11 && (x === 0 || x === size - 1));
      
      const isCenter =
        (x >= 6 && x <= 9 && y >= 6 && y <= 9);

      const isCrosshair =
        (x === 7 || x === 8) || (y === 7 || y === 8);

      if (isCenter || (isCrosshair && x >= 3 && x <= 12 && y >= 3 && y <= 12) || isBorder) {
        // Cyan (#00f0ff): R=0, G=240, B=255, A=255
        buffer[idx] = 0;       // R
        buffer[idx + 1] = 240; // G
        buffer[idx + 2] = 255; // B
        buffer[idx + 3] = 255; // A
      } else {
        // Transparent
        buffer[idx] = 0;
        buffer[idx + 1] = 0;
        buffer[idx + 2] = 0;
        buffer[idx + 3] = 0;
      }
    }
  }

  return nativeImage.createFromBuffer(buffer, { width: size, height: size });
}
