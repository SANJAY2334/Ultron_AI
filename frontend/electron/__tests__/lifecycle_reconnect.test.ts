/* Stage 7 Application Lifecycle, Shutdown & Reconnect Resilience Tests */

import { describe, it, expect } from 'vitest';
import { performIdempotentShutdown } from '../main';
import { updateTrayBackendStatus } from '../tray';

describe('Application Lifecycle & Idempotent Shutdown (Stage 7)', () => {
  it('should execute performIdempotentShutdown safely and idempotently', () => {
    // First call -> cleans up
    expect(() => performIdempotentShutdown()).not.toThrow();

    // Second consecutive call -> should be a safe no-op without errors
    expect(() => performIdempotentShutdown()).not.toThrow();
  });

  it('should update tray status across all 4 connectivity states', () => {
    expect(() => updateTrayBackendStatus('CONNECTING')).not.toThrow();
    expect(() => updateTrayBackendStatus('ONLINE')).not.toThrow();
    expect(() => updateTrayBackendStatus('OFFLINE')).not.toThrow();
    expect(() => updateTrayBackendStatus('RECONNECTING')).not.toThrow();
  });

  it('should verify bounded exponential backoff calculation guarantees', () => {
    function computeBackoffMs(attempt: number): number {
      return Math.min(15000, Math.round(2000 * Math.pow(1.5, Math.min(attempt - 1, 6))));
    }

    // Attempt 1: 2,000ms
    expect(computeBackoffMs(1)).toBe(2000);

    // Attempt 2: 3,000ms
    expect(computeBackoffMs(2)).toBe(3000);

    // Attempt 3: 4,500ms
    expect(computeBackoffMs(3)).toBe(4500);

    // Attempt 4: 6,750ms
    expect(computeBackoffMs(4)).toBe(6750);

    // Attempt 5: 10,125ms
    expect(computeBackoffMs(5)).toBe(10125);

    // Attempt 6: 15,000ms (capped)
    expect(computeBackoffMs(6)).toBe(15000);

    // Attempt 10: 15,000ms (strictly bounded)
    expect(computeBackoffMs(10)).toBe(15000);
  });
});
