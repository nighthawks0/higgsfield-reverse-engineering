// EnhanceComposite — upscale + frame interpolation preview
// Reverse-engineered Higgsfield Video Enhancement (FLUX 3 / ByteDance / Topaz).
//
// Props:
//   inputVideo: input video URL
//   upscaleModel: real-esrgan | byte-dance | topaz
//   fpsBoost: whether to interpolate to 60fps
//
// Approach:
//   - Input video plays at base resolution
//   - Upscale model applied server-side (Modal: Real-ESRGAN / RIFE)
//   - Remotion renders a preview composition at target resolution
//   - Upscaling runs LAST (after content is settled) per Higgsfield guidance

import {
  interpolate,
  useCurrentFrame,
  useVideoConfig,
  Video,
} from "remotion";

export interface EnhanceCompositeProps {
  inputVideo?: string;
  upscaleModel?: string;
  fpsBoost?: boolean;
}

export const EnhanceComposite: React.FC<EnhanceCompositeProps> = ({
  inputVideo = "",
  upscaleModel = "real-esrgan",
  fpsBoost = false,
}) => {
  const { durationInFrames } = useVideoConfig();
  const frame = useCurrentFrame();

  // Slow zoom to demonstrate detail (dev aid; real enhancement is server-side)
  const scale = interpolate(frame, [0, durationInFrames], [1, 1.04]);

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
      {inputVideo && (
        <Video
          resource={inputVideo}
          style={{
            width: "100%",
            height: "100%",
            objectFit: "cover",
            transform: `scale(${scale})`,
          }}
        />
      )}

      {/* Dev overlay: enhancement info */}
      <div
        style={{
          position: "absolute",
          top: 32,
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
        {upscaleModel} {fpsBoost ? "+ 60fps boost" : ""}
      </div>
    </div>
  );
};
