// SoulIdComposite — composite consistent identity across frames
// Reverse-engineered Higgsfield Soul ID.
//
// Props:
//   characterImage: base character image URL
//   referencePhotos: array of reference photo URLs (training set)
//   prompt: optional style/pose direction
//
// Approach:
//   - Base character image forms the anchor
//   - Reference photos are used as visual grounding (IP-Adapter style)
//   - In production: apply a trained LoRA / face-swap (InsightFace/Roop) per frame
//   - Remotion holds the identity stable by keeping the same base image + reference overlay
//
// Soul ID trains once (20+ photos, 3-5 min, 25 credits) and reuses the identity
// across every generation regardless of style/pose/lighting/prompt.

import {
  interpolate,
  useCurrentFrame,
  useVideoConfig,
  Img,
} from "remotion";

export interface SoulIdCompositeProps {
  characterImage?: string;
  referencePhotos?: string[];
  prompt?: string;
}

export const SoulIdComposite: React.FC<SoulIdCompositeProps> = ({
  characterImage = "",
  referencePhotos = [],
  prompt = "",
}) => {
  const { durationInFrames } = useVideoConfig();
  const frame = useCurrentFrame();

  // Cycle through reference photos slowly for visual grounding (dev aid)
  const refIndex = Math.floor(
    interpolate(frame, [0, durationInFrames], [0, referencePhotos.length - 0.001])
  );
  const refPhoto = referencePhotos[Math.min(refIndex, referencePhotos.length - 1)];

  // Gentle identity-stabilizing pulse (subtle)
  const pulse = interpolate(frame, [0, durationInFrames / 2], [1, 1.02]);

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
      {/* Base character image — the consistent identity anchor */}
      {characterImage && (
        <Img
          src={characterImage}
          style={{
            width: "100%",
            height: "100%",
            objectFit: "cover",
            transform: `scale(${pulse})`,
          }}
        />
      )}

      {/* Reference photo overlay (semi-transparent) — grounds identity */}
      {refPhoto && (
        <Img
          src={refPhoto}
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: "100%",
            height: "100%",
            objectFit: "cover",
            opacity: 0.15,
            mixBlendMode: "soft-light",
            pointerEvents: "none",
          }}
        />
      )}

      {/* Dev overlay: prompt */}
      {prompt && (
        <div
          style={{
            position: "absolute",
            bottom: 32,
            left: 32,
            right: 32,
            color: "#fff",
            fontFamily: "system-ui, sans-serif",
            fontSize: 16,
            fontWeight: 600,
            background: "rgba(0,0,0,0.5)",
            padding: "8px 12px",
            borderRadius: 8,
            zIndex: 10,
            textAlign: "center",
          }}
        >
          {prompt}
        </div>
      )}
    </div>
  );
};
