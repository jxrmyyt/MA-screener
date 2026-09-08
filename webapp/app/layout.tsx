import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "M&A Target Screener",
  description: "Cross-industry acquisition-target screener",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
