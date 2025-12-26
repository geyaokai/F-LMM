export type Role = "user" | "assistant" | string;

export interface HistoryItem {
  role: Role;
  text: string;
}

export interface ImageInfo {
  name?: string | null;
  path?: string | null;
  width?: number;
  height?: number;
}

export interface SessionPayload {
  session_id: string;
  session_dir?: string | null;
  session_dir_url?: string | null;
  history: HistoryItem[];
  history_turns?: number;
  image?: ImageInfo | null;
  created_at: string;
  last_active: string;
}

export interface PhraseItem {
  index: number;
  text: string;
  char_span?: [number, number];
  token_span?: [number, number];
}

export interface AskData {
  mode: string;
  answer?: string;
  phrases?: PhraseItem[];
  history: HistoryItem[];
  history_turns?: number;
}

export interface GroundRecord {
  index: number;
  phrase: string;
  overlay_url?: string | null;
  mask_url?: string | null;
  roi_url?: string | null;
  char_span?: [number, number];
  token_span?: [number, number];
  bbox?: number[];
}

export interface GroundData {
  records: GroundRecord[];
  round_dir?: string | null;
  round_url?: string | null;
  history?: HistoryItem[];
}

export interface ApiEnvelope<T> {
  status: string;
  data: T;
  message: string;
}
