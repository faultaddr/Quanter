/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/api/:path*',
      },
    ];
  },
  async headers() {
    return [
      {
        source: '/api/backtest/run-all-stream',
        headers: [
          { key: 'Cache-Control', value: 'no-cache, no-transform' },
          { key: 'X-Accel-Buffering', value: 'no' },
          { key: 'Content-Type', value: 'text/event-stream' },
          { key: 'Connection', value: 'keep-alive' },
          { key: 'Transfer-Encoding', value: 'chunked' },
        ],
      },
    ];
  },
};

module.exports = nextConfig;
