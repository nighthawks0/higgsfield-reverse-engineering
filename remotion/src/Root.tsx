// Higgsfield Reverse-Engineering — Remotion Compositions
// Compositing/rendering layer for the reverse-engineered pipeline.
//
// Stack: Remotion 4.0.532, React 19, Tailwind v4
// Output: 1080x1920 (9:16) short-form reels/shorts/tiktok
//
// Compositions:
//   - GenjutsuComposite: composite depth-masked people over background + audio
//   - LipsyncComposite: composite audio-synced mouth onto character
//   - SoulIdComposite: composite consistent identity across frames
//   - EnhanceComposite: upscale + frame interpolation preview
//
// Render on Modal (see focusgate-reels-modal-20261021/modal_pipeline.py conventions).

import { Composition, registerRoot } from "remotion";
import { GenjutsuComposite } from "./components/GenjutsuComposite";
import { LipsyncComposite } from "./components/LipsyncComposite";
import { SoulIdComposite } from "./components/SoulIdComposite";
import { EnhanceComposite } from "./components/EnhanceComposite";

export const FPS = 30;
export const WIDTH = 1080;
export const HEIGHT = 1920;

export const RemotionRoot = () => (
  <>
    {/* Genjutsu motion-transfer composite — 15s short-form */}
    <Composition
      id="GenjutsuComposite"
      component={GenjutsuComposite}
      fps={FPS}
      width={WIDTH}
      height={HEIGHT}
      durationInFrames={15 * FPS}
      defaultProps={{
        sourceVideo: "",
        depthVideo: "",
        peopleCutout: "",
        referenceImages: [],
        prompt: "",
      }}
    />

    {/* Lipsync studio composite — audio-driven mouth sync */}
    <Composition
      id="LipsyncComposite"
      component={LipsyncComposite}
      fps={FPS}
      width={WIDTH}
      height={HEIGHT}
      durationInFrames={10 * FPS}
      defaultProps={{
        characterImage: "",
        audioUrl: "",
        phonemes: [],
        model: "wan-2.5-speak",
      }}
    />

    {/* Soul ID character consistency composite */}
    <Composition
      id="SoulIdComposite"
      component={SoulIdComposite}
      fps={FPS}
      width={WIDTH}
      height={HEIGHT}
      durationInFrames={10 * FPS}
      defaultProps={{
        characterImage: "",
        referencePhotos: [],
        prompt: "",
      }}
    />

    {/* Video enhancement composite — upscale + interpolation preview */}
    <Composition
      id="EnhanceComposite"
      component={EnhanceComposite}
      fps={FPS}
      width={WIDTH}
      height={HEIGHT}
      durationInFrames={10 * FPS}
      defaultProps={{
        inputVideo: "",
        upscaleModel: "real-esrgan",
        fpsBoost: false,
      }}
    />
  </>
);

registerRoot(RemotionRoot);
