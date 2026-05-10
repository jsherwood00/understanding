// Shared constants and helpers for the /monitor surface — kept in one
// place so the API routes and the client-side dropdown stay in sync.
//
// IMPORTANT: log file paths are *whitelisted* here. The /api/log route
// rejects any ?file=... that isn't in this list. This is the only
// defense against an attacker dragging the route into reading arbitrary
// files on disk; do not weaken it without thinking through traversal
// (.., absolute, symlinks).

import path from "node:path";

const PROJECT_ROOT = process.cwd();

/** Repo-relative log paths. The frontend echoes these strings back as
 *  the `?file=` query param. The route resolves them against
 *  PROJECT_ROOT before any fs access. */
export const ALLOWED_LOG_FILES = [
  "data/no_thinking/extract.log",
  "data/no_thinking/run.log",
  "data/thinking/extract.log",
  "data/thinking/run.log",
  "data/neutral/extract.log",
  "data/neutral/run.log",
] as const;

export type AllowedLogFile = (typeof ALLOWED_LOG_FILES)[number];

export function isAllowedLogFile(p: string): p is AllowedLogFile {
  return (ALLOWED_LOG_FILES as readonly string[]).includes(p);
}

/** Resolve a whitelisted relative log path to an absolute path under
 *  the project root. Throws on anything outside the whitelist. */
export function resolveLogPath(rel: string): string {
  if (!isAllowedLogFile(rel)) {
    throw new Error(`log path not allowed: ${rel}`);
  }
  return path.join(PROJECT_ROOT, rel);
}

/** Corpora that carry an activations/ subdirectory. Used by the storage
 *  endpoint to walk only the directories that matter. */
export const CORPORA = ["no_thinking", "thinking", "neutral"] as const;
export type Corpus = (typeof CORPORA)[number];

export function activationsDir(corpus: Corpus): string {
  return path.join(PROJECT_ROOT, "data", corpus, "activations");
}

/** Hard cap the pipeline must not exceed. Surface to the UI as a
 *  progress-bar denominator. */
export const STORAGE_CAP_BYTES = 200 * 1024 * 1024 * 1024;

/** Flag files the pipeline reads/writes. We only ever read the thermal
 *  one and read/write the user one. */
export const FLAG_THERMAL_PAUSE = "/tmp/understanding_pause_thermal";
export const FLAG_USER_PAUSE = "/tmp/understanding_pause_user";
