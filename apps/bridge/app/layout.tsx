import type { Metadata, Viewport } from "next";
import "./globals.css";

// No `next/font/google`. The scaffold fetches Geist from Google's CDN at build
// time, which makes an offline clone fail to build and adds a third-party
// request to a page that otherwise has none. A judge should be able to clone
// this repository on hotel wifi and run it. System fonts cost nothing here --
// this is an instrument panel, not a brand surface.

export const metadata: Metadata = {
  title: "Marine-AI — Simulator Console",
  description:
    "Retrofittable IoT and AI advisory system for Philippine diesel passenger boats. " +
    "Advisory only; the captain retains command.",
};

export const viewport: Viewport = {
  themeColor: "#020617",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full">{children}</body>
    </html>
  );
}
