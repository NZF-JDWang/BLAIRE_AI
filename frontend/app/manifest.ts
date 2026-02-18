import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "BLAIRE",
    short_name: "BLAIRE",
    description: "Blacksite Lab AI Hub",
    start_url: "/",
    display: "standalone",
    background_color: "#f5f7fa",
    theme_color: "#0f172a",
    icons: [
      {
        src: "/icon.svg",
        type: "image/svg+xml",
        sizes: "any",
      },
    ],
  };
}
