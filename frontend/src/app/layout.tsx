import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "NEXUS-AI | Enterprise ML Platform",
  description:
    "Production-grade AI platform with real-time fraud detection, hybrid recommendation engine, NLP sentiment pipeline, and RAG-powered document intelligence.",
  keywords: ["AI", "Machine Learning", "Fraud Detection", "Recommendations", "NLP", "Enterprise"],
};

import { AuthProvider } from "@/context/AuthContext";

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="dark">
      <head>
        <meta name="theme-color" content="#05050f" />
      </head>
      <body className={`${inter.variable} antialiased`}>
        <AuthProvider>
          {children}
        </AuthProvider>
      </body>
    </html>
  );
}
