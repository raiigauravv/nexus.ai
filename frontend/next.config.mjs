/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    // ESLint errors won't block the production build (still shown in dev)
    ignoreDuringBuilds: true,
  },
  typescript: {
    // Allow production builds to succeed even with type errors
    ignoreBuildErrors: true,
  },
};

export default nextConfig;
