/* ULTRON Electron Whitelisted IPC Channels (Phase 4G.4: Stage 3)
 *
 * Authoritative list of whitelisted IPC channels.
 *
 * HARD SECURITY INVARIANTS:
 * - Only strictly declared channels are registered or allowed to communicate.
 * - No dynamic/arbitrary channel names.
 */

export const WHITELISTED_IPC_CHANNELS = [
  'ultron:window:minimize',
  'ultron:window:maximize',
  'ultron:window:close-to-tray',
  'ultron:hud:toggle',
  'ultron:notification:send',
  'ultron:shortcut:activated',
  'ultron:backend:status',
  'ultron:platform:get-info',
] as const;

export type WhitelistedIPCChannel = (typeof WHITELISTED_IPC_CHANNELS)[number];
