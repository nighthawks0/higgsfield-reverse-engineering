// GenjutsuComposite — composite depth-masked people over background + audio
// Reverse-engineered Higgsfield Genjutsu motion transfer.
//
// Props:
//   sourceVideo: original video URL (publicly accessible)
//   depthVideo: depth-masked video URL (Depth Anything output)
//   peopleCutout: people cut out on transparent background (SAM output)
//   referenceImages: public image URLs for new cast/wardrobe
//   prompt: optional description of the change
//
// Layers (bottom to top):
//   1. sourceVideo — original background/motion reference
//   2. depthVideo — depth mask (keeps motion, throws away appearance)
//   3. peopleCutout — cut-out people composited over background (stops model tracing outline)
//   4. referenceImages overlay (optional, for visual grounding)
// Audio: from sourceVideo (original soundtrack restored at final step)

import { Sequence, useVideoConfig, Video, Img } from "remotion";

export interface GenjutsuCompositeProps {
  sourceVideo?: string;
  depthVideo?: string;
  peopleCutout?: string;
  referenceImages?: string[];
  prompt?: string;
}

export const GenjutsuComposite: React.FC<GenjutsuCompositeProps> = ({
  sourceVideo = "",
  depthVideo = "",
  peopleCutout = "",
  referenceImages = [],
  prompt = "",
}) => {
  const { durationInFrames } = useVideoConfig();

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
      {/* Layer 1: original video — background + motion reference */}
      {sourceVideo && (
        <Video
          resource={sourceVideo}
          style={{ width: "100%", height: "100%", objectFit: "cover" }}
        />
      )}

      {/* Layer 2: depth mask — semi-transparent, shows motion without appearance */}
      {depthVideo && (
        <Video
          resource={depthVideo}
          style={{
            width: "100%",
            height: "100%",
            objectFit: "cover",
            opacity: 0.35,
            mixBlendMode: "luminosity",
          }}
        />
      )}

      {/* Layer 3: people cutout — cut-out people over original background */}
      {peopleCutout && (
        <Video
          resource={peopleCutout}
          style={{
            width: "100%",
            height: "100%",
            objectFit: "cover",
            opacity: 1,
          }}
        />
      )}

      {/* Layer 4: reference images (optional) — shown briefly at start for grounding */}
      {referenceImages.length > 0 && (
        <Sequence from={0} durationInFrames={Math.min(90, durationInFrames)}>
          <Img
            src={referenceImages[0]}
            alt="reference"
            style={{
              position: "absolute",
              top: 16,
              left: 16,
              width: 120,
              height: 120,
              objectFit: "cover",
              borderRadius: 16,
              border: "2px solid rgba(255,255,255,0.6)",
              boxShadow: "0 4px 12px rgba(0,0,0,0.4)",
              zIndex: 10,
            }}
          />
        </Sequence>
      )}

      {/* Prompt overlay (debug/dev only) */}
      {prompt && (
        <div
          style={{
            position: "absolute",
            bottom: 32,
            left: 32,
            right: 32,
            color: "#fff",
            fontFamily: "system-ui, sans-serif",
            fontSize: 18,
            fontWeight: 600,
            background: "rgba(0,0,0,0.5)",
            padding: "10px 14px",
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
