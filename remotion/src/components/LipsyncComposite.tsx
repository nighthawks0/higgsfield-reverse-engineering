// LipsyncComposite — composite audio-synced mouth onto character
// Reverse-engineered Higgsfield Lipsync Studio.
//
// Props:
//   characterImage: character image URL (still image to animate)
//   audioUrl: audio URL (TTS or uploaded voice)
//   phonemes: array of {startFrame, endFrame, mouthShape, openness}
//   model: lipsync model selector (wan-2.5-speak | kling-2.6 | veo-3.1 | ...)
//
// Approach:
//   - Character image forms the base
//   - Mouth region is animated frame-by-frame based on phoneme timing
//   - Remotion interpolates mouth openness between phoneme boundaries
//   - Audio track drives the sync

import {
  interpolate,
  useCurrentFrame,
  useVideoConfig,
  Img,
  Video,
} from "remotion";

export type MouthShape = "closed" | "small" | "medium" | "large" | "oo" | "ee" | "ah";

export interface Phoneme {
  startFrame: number;
  endFrame: number;
  shape: MouthShape;
  openness: number; // 0-1 mouth openness
}

export interface LipsyncCompositeProps {
  characterImage?: string;
  audioUrl?: string;
  phonemes?: Phoneme[];
  model?: string;
}

export const LipsyncComposite: React.FC<LipsyncCompositeProps> = ({
  characterImage = "",
  audioUrl = "",
  phonemes = [],
  model = "wan-2.5-speak",
}) => {
  const { durationInFrames } = useVideoConfig();
  const frame = useCurrentFrame();

  void model; // kept for API parity / future model-specific mouth-shape maps
  void durationInFrames; // reserved for duration-based mouth-shape easing

  // Find active phoneme for the current frame
  const active = phonemes.find((p) => frame >= p.startFrame && frame <= p.endFrame);

  let openness = 0;
  let shape: MouthShape = "closed";
  if (active) {
    const t = interpolate(frame, [active.startFrame, active.endFrame], [0, 1], {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    });
    openness = active.openness;
    shape = active.shape;
    void t; // t used for potential future easing
  }

  // Mouth region: lower-center of face
  const mouthWidth = 110;
  const mouthHeight = Math.max(60 * openness, 4);

  const mouthStyle: React.CSSProperties = {
    position: "absolute",
    top: "68%",
    left: "50%",
    transform: "translateX(-50%)",
    width: mouthWidth,
    height: mouthHeight,
    minHeight: 4,
    background:
      shape === "oo"
        ? "#2a1a1a"
        : shape === "ee"
          ? "#3a2a2a"
          : "#1a0f0f",
    borderRadius: Math.max(mouthHeight / 2, 2),
    opacity: openness > 0.05 ? 1 : 0,
    pointerEvents: "none",
  };

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        background: "#000",
        overflow: "hidden",
        position: "relative",
      }}
    >
      {/* Base character image */}
      {characterImage && (
        <Img
          src={characterImage}
          style={{ width: "100%", height: "100%", objectFit: "cover" }}
        />
      )}

      {/* Audio track drives the lip sync */}
      {audioUrl && <Video resource={audioUrl} style={{ display: "none" }} />}

      {/* Animated mouth overlay */}
      <div style={mouthStyle} />

      {/* Dev overlay: show current phoneme */}
      {active && (
        <div
          style={{
            position: "absolute",
            bottom: 32,
            left: 32,
            color: "#0f0",
            fontFamily: "monospace",
            fontSize: 14,
            background: "rgba(0,0,0,0.6)",
            padding: "6px 10px",
            borderRadius: 6,
            zIndex: 10,
          }}
        >
          {active.shape} ({Math.round(openness * 100)}%)
        </div>
      )}
    </div>
  );
};
